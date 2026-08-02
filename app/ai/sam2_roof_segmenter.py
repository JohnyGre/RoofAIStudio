"""
SAM 2 Roof Segmenter - zero-shot roof plane segmentation with SAM 2.

Supports the same modes as SAMRoofSegmenter:
1. Point-click: User clicks a point on the roof → SAM 2 segments that plane
2. Box-prompt: Segment from a bounding box prompt
3. YOLO-assisted: YOLO proposes candidate boxes → SAM 2 refines the masks

Key improvements over SAM 1:
- Better mask quality, especially on fine structures (edges, chimneys, gutters)
- More stable masks across similar prompts
- Slightly faster inference per mask after image encoding

Dependencies: sam2, ultralytics, opencv-python, numpy, torch
"""

import time
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SAM 2 imports (optional — gracefully degrade if not installed)
# ---------------------------------------------------------------------------
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False
    build_sam2 = None  # type: ignore
    SAM2ImagePredictor = None  # type: ignore
    torch = None  # type: ignore


class SAM2RoofSegmenter:
    """
    Zero-shot roof segmentation using Segment Anything Model 2 (SAM 2).

    API-compatible replacement for SAMRoofSegmenter.
    """

    MODEL_VARIANTS = {
        "tiny": {
            "config": "sam2_hiera_t.yaml",
            "checkpoint": "sam2_hiera_tiny.pt",
            "size_mb": 155,
            "speed": "fastest",
        },
        "small": {
            "config": "sam2_hiera_s.yaml",
            "checkpoint": "sam2_hiera_small.pt",
            "size_mb": 176,
            "speed": "fast",
        },
        "base_plus": {
            "config": "sam2_hiera_b+.yaml",
            "checkpoint": "sam2_hiera_base_plus.pt",
            "size_mb": 324,
            "speed": "medium",
        },
        "large": {
            "config": "sam2_hiera_l.yaml",
            "checkpoint": "sam2_hiera_large.pt",
            "size_mb": 224,
            "speed": "slow",
        },
    }

    def __init__(
        self,
        model_type: str = "small",
        yolo_model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if not HAS_SAM2:
            raise ImportError(
                "sam2 is not installed. Install with:\n"
                "  pip install sam2\n"
                "For full install instructions see: "
                "https://github.com/facebookresearch/segment-anything-2"
            )

        if model_type not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown SAM 2 variant '{model_type}'. "
                f"Choose from: {list(self.MODEL_VARIANTS.keys())}"
            )

        self.model_type = model_type
        self.yolo_model_path = yolo_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sam_model = None
        self._predictor = None
        self._yolo = None
        self._loaded = False
        self._current_image_bgr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        sam_checkpoint: Optional[str] = None,
        model_cfg: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
    ) -> bool:
        """
        Load SAM 2 and optionally YOLO models.

        Args:
            sam_checkpoint: Path to .pt checkpoint. If None, uses variant default.
            model_cfg: Path to model config YAML. If None, uses variant default.
            yolo_model_path: Optional YOLO model for auto mode.
        """
        variant = self.MODEL_VARIANTS[self.model_type]

        # Resolve checkpoint path
        if sam_checkpoint is None:
            sam_checkpoint = f"ai_models/{variant['checkpoint']}"
        ckpt_path = Path(sam_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not found: {ckpt_path}\n"
                f"Download it from https://github.com/facebookresearch/segment-anything-2"
            )

        # Resolve config path
        if model_cfg is None:
            # Try common locations
            cfg_name = variant["config"]
            cfg_candidates = [
                Path("config") / cfg_name,
                Path("ai_models") / cfg_name,
                # sam2 package configs
                Path(__import__("sam2").__file__).parent / "configs" / cfg_name,
            ]
            cfg_path = None
            for c in cfg_candidates:
                if c.exists():
                    cfg_path = str(c)
                    break
            if cfg_path is None:
                raise FileNotFoundError(
                    f"SAM 2 config '{cfg_name}' not found. Searched:\n"
                    + "\n".join(f"  {c}" for c in cfg_candidates)
                )
        else:
            cfg_path = model_cfg

        logger.info(
            f"Loading SAM 2 ({self.model_type}) from {ckpt_path} "
            f"on {self.device}"
        )
        t0 = time.time()

        self._sam_model = build_sam2(
            cfg_path,
            str(ckpt_path),
            device=self.device,
        )
        self._sam_model.eval()
        self._predictor = SAM2ImagePredictor(self._sam_model)

        logger.info(
            f"SAM 2 loaded ({time.time() - t0:.1f}s, "
            f"{variant['size_mb']} MB)"
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

    def _set_image_once(self, image_bgr: np.ndarray):
        """
        Encode image with SAM 2 if it changed.
        SAM 2 expects RGB uint8 numpy array.
        """
        # Quick identity check — same object or identical content
        if self._current_image_bgr is image_bgr:
            return
        if (
            self._current_image_bgr is not None
            and self._current_image_bgr.shape == image_bgr.shape
            and np.array_equal(self._current_image_bgr, image_bgr)
        ):
            return

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._predictor.set_image(image_rgb)
        self._current_image_bgr = image_bgr

    @staticmethod
    def _extract_contours_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Return (largest_contour, area) from a bool/uint8 mask."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.array([]), 0
        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        return best, int(area)

    @staticmethod
    def _simplify_contour(contour: np.ndarray, epsilon_ratio: float = 0.003) -> np.ndarray:
        if contour.size == 0:
            return contour
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    def _predict_mask(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> Tuple[List[np.ndarray], List[float], Any]:
        """
        Wrapper around SAM 2 predictor with proper autocast for speed.
        """
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self._predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=box,
                        multimask_output=multimask_output,
                    )
            else:
                masks, scores, logits = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask_output,
                )
        return masks, scores, logits

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
        self._set_image_once(image)

        t0 = time.time()
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 = foreground

        masks, scores, _ = self._predict_mask(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=multimask,
        )

        results = []
        for mask, score in zip(masks, scores):
            contour, area = self._extract_contours_from_mask(mask)
            if area < 10:
                continue

            approx = self._simplify_contour(contour)
            results.append(
                {
                    "mask": mask,
                    "score": float(score),
                    "area": area,
                    "contour": approx,
                    "sam_time_ms": (time.time() - t0) * 1000,
                }
            )

        logger.info(
            f"SAM2 point ({x},{y}): {len(results)} masks in "
            f"{(time.time()-t0)*1000:.0f}ms"
        )
        return results

    # ------------------------------------------------------------------
    # Mode 1b: Box-prompt segmentation
    # ------------------------------------------------------------------

    def segment_from_box(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> List[Dict[str, Any]]:
        """
        Segment a roof plane from a bounding box prompt.

        Args:
            image: BGR image
            x1, y1, x2, y2: Bounding box coordinates
        """
        self._ensure_loaded()
        self._set_image_once(image)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        t0 = time.time()
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = self._predict_mask(
            box=input_box[None, :],
            multimask_output=False,
        )

        results = []
        for mask, score in zip(masks, scores):
            contour, area = self._extract_contours_from_mask(mask)
            if area < 50:
                continue

            approx = self._simplify_contour(contour, epsilon_ratio=0.003)
            results.append(
                {
                    "mask": mask,
                    "score": float(score),
                    "area": area,
                    "contour": approx,
                    "box": (x1, y1, x2, y2),
                    "sam_time_ms": (time.time() - t0) * 1000,
                }
            )

        logger.info(
            f"SAM2 box ({x1},{y1},{x2},{y2}): {len(results)} masks in "
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
        Full-auto: YOLO finds candidate boxes → SAM 2 refines each.

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

        # Step 2: SAM 2 encodes image once, then refines each box
        self._set_image_once(img_640)

        sam_t0 = time.time()
        results = []

        for i, box in enumerate(boxes):
            cls_id = int(yolo_results.boxes.cls[i])
            cls_name = self._yolo.names.get(cls_id, f"class_{cls_id}")
            yolo_conf = float(yolo_results.boxes.conf[i])

            try:
                masks, scores, _ = self._predict_mask(
                    box=box[None, :],
                    multimask_output=False,
                )
            except Exception as e:
                logger.warning(f"SAM2 predict failed for box {i}: {e}")
                continue

            if len(masks) == 0:
                continue

            mask = masks[0]
            sam_score = float(scores[0]) if len(scores) > 0 else 0.0
            area = int(np.sum(mask))

            if area < min_mask_area:
                continue

            contour, _ = self._extract_contours_from_mask(mask)
            if contour.size == 0:
                continue

            approx = self._simplify_contour(contour, epsilon_ratio=0.004)

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

        for r in results:
            r["sam_time_ms"] = per_box

        logger.info(
            f"SAM2 refined: {len(results)}/{len(boxes)} boxes in "
            f"{sam_total:.1f}s ({per_box:.0f}ms each)"
        )
        logger.info(f"Total: {(time.time()-t0)*1000:.0f}ms")

        return results

    # ------------------------------------------------------------------
    # Visualization helpers (identical API to SAMRoofSegmenter)
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
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self):
        """Free SAM 2 / YOLO from memory."""
        self._current_image_bgr = None
        del self._sam_model
        del self._predictor
        del self._yolo
        self._sam_model = None
        self._predictor = None
        self._yolo = None
        self._loaded = False
        logger.info("SAM 2 segmenter unloaded")
