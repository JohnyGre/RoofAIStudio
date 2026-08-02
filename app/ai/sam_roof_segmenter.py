"""
SAM Roof Segmenter - zero-shot roof plane segmentation for any view angle.

Supports two modes:
1. Point-click: User clicks a point on the roof → SAM segments that plane
2. YOLO-assisted: YOLO proposes candidate boxes → SAM refines the masks

Works on satellite, aerial, drone, and street-view photos without retraining.

Dependencies: segment-anything, ultralytics, opencv-python, numpy
"""

import time
import logging
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SAMRoofSegmenter:
    """
    Zero-shot roof segmentation using Segment Anything Model (SAM).

    Key advantage: No dataset needed for side-view (street) photos.
    YOLO provides candidate boxes; SAM refines them into precise masks.

    Usage:
        seg = SAMRoofSegmenter()
        seg.load()

        # Mode 1: click-based segmentation
        mask = seg.segment_from_point(image, x=320, y=240)

        # Mode 2: YOLO-assisted full-auto
        results = seg.segment_auto(image, conf=0.25)
        for r in results:
            print(r["mask"], r["class"], r["confidence"])
    """

    MODEL_VARIANTS = {
        "vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "size_mb": 375,
            "speed": "fast",
        },
        "vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "size_mb": 1250,
            "speed": "medium",
        },
        "vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "size_mb": 2560,
            "speed": "slow",
        },
    }

    def __init__(
        self,
        model_type: str = "vit_b",
        yolo_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.yolo_model_path = yolo_model_path
        self.device = device
        self._sam = None
        self._predictor = None
        self._yolo = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        sam_checkpoint: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
    ) -> bool:
        """Load SAM and optionally YOLO models."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment-anything not installed. Run: pip install segment-anything"
            )

        if sam_checkpoint is None:
            sam_checkpoint = f"ai_models/sam_{self.model_type}_01ec64.pth"

        # Load SAM
        logger.info(f"Loading SAM {self.model_type} from {sam_checkpoint}")
        t0 = time.time()

        model_fn = sam_model_registry[self.model_type]
        self._sam = model_fn(checkpoint=sam_checkpoint)
        self._sam.to(self.device)
        self._sam.eval()
        self._predictor = SamPredictor(self._sam)

        logger.info(
            f"SAM loaded ({time.time() - t0:.1f}s, "
            f"{self.MODEL_VARIANTS[self.model_type]['size_mb']} MB)"
        )

        # Optionally load YOLO
        yolo_path = yolo_model_path or self.yolo_model_path
        if yolo_path:
            try:
                from ultralytics import YOLO

                self._yolo = YOLO(yolo_path)
                self._yolo.to(self.device)
                logger.info(f"YOLO loaded: {yolo_path}")
            except ImportError:
                logger.warning("ultralytics not installed; YOLO-assisted mode disabled")
            except Exception as e:
                logger.warning(f"Failed to load YOLO: {e}")

        self._loaded = True
        return True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def has_yolo(self) -> bool:
        return self._yolo is not None

    # ------------------------------------------------------------------
    # Mode 1: Point-click segmentation
    # ------------------------------------------------------------------

    def segment_from_point(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        multimask: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Segment a roof plane from a single click point.

        Args:
            image: BGR image (H, W, 3)
            x, y: Click coordinates
            multimask: If True, return 3 candidate masks (different granularities)

        Returns:
            List of dicts: [{
                "mask": np.ndarray (H, W) bool,
                "score": float (SAM confidence),
                "area": int (pixels),
                "contour": np.ndarray (N, 1, 2) polygon,
            }, ...]
        """
        self._ensure_loaded()
        h, w = image.shape[:2]

        # SAM: set image context
        self._predictor.set_image(image)

        # Predict
        t0 = time.time()
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self._predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask,
        )

        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # Use largest contour
            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)

            # Simplify polygon
            epsilon = 0.003 * cv2.arcLength(best, True)
            approx = cv2.approxPolyDP(best, epsilon, True)

            results.append(
                {
                    "mask": mask,
                    "score": float(score),
                    "area": int(area),
                    "contour": approx,
                    "sam_time_ms": (time.time() - t0) * 1000,
                }
            )

        if not results:
            logger.info(f"SAM point ({x},{y}): no mask found")

        logger.info(
            f"SAM point ({x},{y}): {len(results)} masks in "
            f"{(time.time()-t0)*1000:.0f}ms"
        )
        return results

    def segment_from_box(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> List[Dict[str, Any]]:
        """
        Segment a roof plane from a bounding box.

        Args:
            image: BGR image
            x1, y1, x2, y2: Bounding box coordinates
        """
        self._ensure_loaded()
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        self._predictor.set_image(image)

        t0 = time.time()
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = self._predictor.predict(
            box=input_box[None, :],
            multimask_output=False,
        )

        results = []
        for mask, score in zip(masks, scores):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            best = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best)
            if area < 50:
                continue

            epsilon = 0.003 * cv2.arcLength(best, True)
            approx = cv2.approxPolyDP(best, epsilon, True)

            results.append(
                {
                    "mask": mask,
                    "score": float(score),
                    "area": int(area),
                    "contour": approx,
                    "box": (x1, y1, x2, y2),
                    "sam_time_ms": (time.time() - t0) * 1000,
                }
            )

        logger.info(
            f"SAM box ({x1},{y1},{x2},{y2}): {len(results)} masks in "
            f"{(time.time()-t0)*1000:.0f}ms"
        )
        return results

    # ------------------------------------------------------------------
    # Mode 2: YOLO-assisted auto segmentation
    # ------------------------------------------------------------------

    def segment_auto(
        self,
        image: np.ndarray,
        conf: float = 0.20,
        iou: float = 0.45,
        imgsz: int = 640,
        min_mask_area: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        Full-auto: YOLO finds candidate boxes → SAM refines each.

        Args:
            image: BGR image (any size; auto-resized to imgsz for YOLO step)
            conf: YOLO confidence threshold
            iou: YOLO NMS IoU threshold
            imgsz: YOLO inference size (best at model training size, usually 640)
            min_mask_area: Minimum SAM mask area to keep (filters noise)

        Returns:
            List of dicts, each with mask, class, confidence, contour, etc.
        """
        if not self.has_yolo:
            raise RuntimeError(
                "YOLO model not loaded. Call load() with yolo_model_path."
            )

        self._ensure_loaded()
        orig_h, orig_w = image.shape[:2]

        # Resize to model-native size for YOLO
        t0 = time.time()
        img_640 = cv2.resize(image, (imgsz, imgsz))
        scale_x = orig_w / imgsz
        scale_y = orig_h / imgsz

        # Step 1: YOLO on 640x640
        yolo_results = self._yolo(
            img_640, conf=conf, iou=iou, imgsz=imgsz,
            device=self.device, verbose=False
        )[0]
        yolo_time = time.time() - t0

        if len(yolo_results.boxes) == 0:
            logger.info(f"YOLO: 0 boxes found (conf={conf})")
            return []

        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        logger.info(
            f"YOLO: {len(boxes)} boxes in {yolo_time*1000:.0f}ms (conf={conf})"
        )

        # Step 2: SAM encodes image once, then refines each box
        self._predictor.set_image(img_640)

        sam_t0 = time.time()
        results = []

        for i, box in enumerate(boxes):
            cls_id = int(yolo_results.boxes.cls[i])
            cls_name = self._yolo.names.get(cls_id, f"class_{cls_id}")
            yolo_conf = float(yolo_results.boxes.conf[i])

            try:
                masks, scores, _ = self._predictor.predict(
                    box=box[None, :],
                    multimask_output=False,
                )
            except Exception as e:
                logger.warning(f"SAM predict failed for box {i}: {e}")
                continue

            if len(masks) == 0:
                continue

            mask = masks[0]
            sam_score = float(scores[0]) if len(scores) > 0 else 0.0
            area = int(np.sum(mask))

            if area < min_mask_area:
                continue

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            best_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.004 * cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)

            # Scale contour back to original image coordinates
            contour_scaled = (approx.astype(np.float32) * [scale_x, scale_y]).astype(
                np.int32
            )

            results.append(
                {
                    "mask_640": mask,
                    "score": yolo_conf,
                    "sam_score": sam_score,
                    "area_640": area,
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "contour": contour_scaled,  # original coords
                    "box_orig": (
                        int(box[0] * scale_x),
                        int(box[1] * scale_y),
                        int(box[2] * scale_x),
                        int(box[3] * scale_y),
                    ),
                    "sam_time_ms": 0,  # filled below
                }
            )

        sam_total = time.time() - sam_t0
        per_box = sam_total / max(len(boxes), 1) * 1000

        # Fill per-box timing
        for r in results:
            r["sam_time_ms"] = per_box

        logger.info(
            f"SAM refined: {len(results)}/{len(boxes)} boxes in "
            f"{sam_total:.1f}s ({per_box:.0f}ms each)"
        )
        logger.info(f"Total: {(time.time()-t0)*1000:.0f}ms")

        return results

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    COLORS = [
        (0, 255, 0), (255, 140, 0), (0, 220, 255), (255, 0, 220),
        (200, 255, 0), (255, 200, 0), (0, 255, 200), (255, 120, 200),
        (150, 255, 100), (255, 255, 0), (100, 200, 255), (255, 100, 100),
        (50, 255, 150), (255, 80, 255), (80, 255, 255), (255, 180, 50),
    ]

    def draw_results(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        alpha: float = 0.35,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Draw segmentation results on a copy of the image."""
        vis = image.copy()

        for i, r in enumerate(results):
            color = self.COLORS[i % len(self.COLORS)]
            contour = r["contour"]

            cv2.polylines(vis, [contour], True, color, 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [contour], color)
            vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

            if show_labels:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    label = f"{r.get('class_name','?')} {r.get('score',0):.2f}"
                    cv2.putText(
                        vis, label, (cx - 22, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    cv2.putText(
                        vis, label, (cx - 22, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA,
                    )

        return vis

    def draw_point_result(
        self,
        image: np.ndarray,
        point_results: List[Dict[str, Any]],
        click_point: Tuple[int, int],
        alpha: float = 0.40,
    ) -> np.ndarray:
        """Draw point-click segmentation results."""
        vis = image.copy()
        x, y = click_point

        for i, r in enumerate(point_results):
            color = self.COLORS[i % len(self.COLORS)]
            contour = r["contour"]

            cv2.polylines(vis, [contour], True, color, 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [contour], color)
            vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Draw click marker
        cv2.circle(vis, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)

        return vis

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

    def unload(self):
        """Free SAM/YOLO from memory."""
        del self._sam
        del self._predictor
        del self._yolo
        self._sam = None
        self._predictor = None
        self._yolo = None
        self._loaded = False
        logger.info("SAM segmenter unloaded")
