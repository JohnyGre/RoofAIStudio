"""
Hybrid roof detector v2 — uses YOLO building segmentation model + SAHI tiling
+ multi-strategy OpenCV contours + roof-specific color/texture filtering.

Architecture:
  1. YOLOv8n-building-seg → segmentace budov (model natrenovany na budovy)
  2. SAHI tiling → pro velke ortofotomapy (4K+)
  3. Multi-strategy OpenCV → konturova detekce (edge/watershed/k-means/morph)
  4. Roof color/texture classifier → filtruje travu, oblohu, stromy
  5. Merge & NMS dedup → kombinuje vsechny zdroje
"""

import uuid
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from app.ai.ai_result import DetectionResult, BBoxGeometry
from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector, ContourRefiner
from app.ai.sahi_detector import SAHIDetector, TilePrediction
from app.core.logger import setup_logging
from app.geometry.point import Point2D

logger = setup_logging()

# Try to import ultralytics (optional dependency)
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    YOLO = None
    HAS_ULTRALYTICS = False


@dataclass
class RoofRegion:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    confidence: float
    source: str
    color_score: float = 0.0
    texture_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoofColorClassifier:
    """Classifies image regions as roof-like based on color and texture."""

    ROOF_COLORS = {
        "red_clay":      ((0, 50, 100), (80, 160, 255)),
        "brown_shingle": ((20, 60, 80), (120, 160, 200)),
        "gray_metal":    ((60, 60, 60), (200, 200, 200)),
        "dark_gray":     ((20, 20, 20), (100, 100, 100)),
        "terracotta":    ((0, 40, 120), (60, 130, 240)),
        "green_metal":   ((30, 80, 20), (120, 200, 120)),
        "blue_metal":    ((50, 0, 0), (255, 80, 120)),
        "beige_tile":    ((0, 100, 100), (80, 220, 220)),
    }

    @classmethod
    def score_roof_color(cls, region_pixels: np.ndarray) -> float:
        if region_pixels.size == 0:
            return 0.0
        mean_bgr = region_pixels.mean(axis=0)
        std_bgr = region_pixels.std(axis=0)

        # Reject bright white / sky blue / grass green
        if mean_bgr[0] > 200 and mean_bgr[1] > 200 and mean_bgr[2] > 200: # Almost white
            return 0.0
        if mean_bgr[0] > mean_bgr[2] and mean_bgr[1] > mean_bgr[2]: # Sky blue / cyan
            return 0.0
        # Aggressively reject green/yellow-green (grass, trees)
        # if green is dominant and not balanced by red/blue
        if mean_bgr[1] > mean_bgr[0] * 1.1 and mean_bgr[1] > mean_bgr[2] * 1.1:
            return 0.01 # Very low score instead of 0.0 to allow for borderline cases

        saturation = float(std_bgr.mean())
        if saturation < 5: # Very low saturation (grayscale)
            return 0.01
        if saturation > 80: # Very high saturation (unnatural)
            return 0.2

        for (low, high) in cls.ROOF_COLORS.values():
            if (low[0] <= mean_bgr[0] <= high[0] and
                low[1] <= mean_bgr[1] <= high[1] and
                low[2] <= mean_bgr[2] <= high[2]):
                return 0.8

        if 8 <= saturation <= 60:
            return 0.7
        elif 5 <= saturation <= 8:
            return 0.3
        return 0.1

    @classmethod
    def score_roof_texture(cls, region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
        if gray.size < 100:
            return 0.0
        edges = cv2.Canny(gray, 30, 90)
        edge_density = float(np.mean(edges > 0))
        if 0.02 <= edge_density <= 0.25:
            return 0.7
        elif edge_density < 0.02:
            return 0.2
        return 0.3


class HybridRoofDetector:
    """
    Hybrid roof detector combining YOLO (building-specific) + SAHI + OpenCV.
    """

    # Default model paths
    BUILDING_MODEL = "ai_models/yolov8n_building_seg.pt"
    COCO_MODEL = "ai_models/yolov8n-seg.pt"

    def __init__(self):
        self._yolo_model = None  # Direct ultralytics YOLO instance
        self._model_type = None  # "building" or "coco" or None
        self._sahi: Optional[SAHIDetector] = None
        self._opencv_detector = RoofPlaneDetector()
        self._yolo_loaded = False
        logger.info("HybridRoofDetector v2 initialized.")

    def load_yolo(
        self,
        model_path: str = BUILDING_MODEL,
        prefer_building_model: bool = True,
    ) -> bool:
        """Load a YOLO model. Prefers building-specific model if available."""
        if not HAS_ULTRALYTICS:
            logger.warning("Ultralytics YOLO not installed. OpenCV only.")
            return False

        # Try building model first
        paths_to_try = []
        if prefer_building_model:
            paths_to_try = [model_path, self.BUILDING_MODEL, self.COCO_MODEL]
        else:
            paths_to_try = [model_path, self.COCO_MODEL]

        for path in paths_to_try:
            if not Path(path).exists():
                continue
            try:
                self._yolo_model = YOLO(path)
                self._yolo_loaded = True
                # Determine model type from class names
                names = self._yolo_model.names
                has_building = "Building" in names.values() or "building" in names.values()
                self._model_type = "building" if has_building else "coco"
                logger.info(
                    f"YOLO loaded: {path} ({self._model_type}, "
                    f"{len(names)} classes)"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.warning("No YOLO model loaded. Using OpenCV only.")
        return False

    def _yolo_detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.2,
        use_sahi: bool = False,
        sahi_tile_size: int = 640,
    ) -> List[RoofRegion]:
        """Run YOLO detection (direct or SAHI-tiled). Returns RoofRegions."""
        regions: List[RoofRegion] = []
        if not self._yolo_loaded or self._yolo_model is None:
            return regions

        h, w = image.shape[:2]
        should_tile = use_sahi and (w > sahi_tile_size * 1.5 or h > sahi_tile_size * 1.5)

        try:
            if should_tile:
                logger.debug(f"Using SAHI tiling for {w}x{h} image")
                self._sahi = SAHIDetector(
                    self._yolo_model,
                    tile_size=sahi_tile_size,
                    overlap_ratio=0.2,
                    conf_threshold=conf_threshold,
                )
                preds = self._sahi.detect(image)
                for p in preds:
                    x1, y1, x2, y2 = p.bbox
                    bw, bh = x2 - x1, y2 - y1
                    area = bw * bh
                    if area < 100:
                        continue
                    pts = np.array([
                        [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
                    ], dtype=np.int32)
                    img_crop = image[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else image[:1, :1]
                    color_score = RoofColorClassifier.score_roof_color(
                        img_crop.reshape(-1, 3)
                    )
                    texture_score = RoofColorClassifier.score_roof_texture(img_crop)
                    regions.append(RoofRegion(
                        contour=pts, bbox=(x1, y1, bw, bh), area=area,
                        confidence=p.confidence, source=f"yolo_{p.class_name}",
                        color_score=color_score, texture_score=texture_score,
                    ))
            else:
                results = self._yolo_model.predict(
                    image, conf=conf_threshold, iou=0.5, verbose=False,
                )
                for r in results:
                    if r.boxes is None:
                        continue
                    for i, box in enumerate(r.boxes):
                        cls_id = int(box.cls.item())
                        cls_name = self._yolo_model.names.get(cls_id, f"c{cls_id}")
                        conf = float(box.conf.item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        bw, bh = x2 - x1, y2 - y1
                        area = bw * bh
                        if area < 200:
                            continue
                        pts = np.array([
                            [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
                        ], dtype=np.int32)
                        img_crop = image[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                        color_score = RoofColorClassifier.score_roof_color(
                            img_crop.reshape(-1, 3) if img_crop.size > 0 else np.zeros((1,3))
                        )
                        texture_score = RoofColorClassifier.score_roof_texture(img_crop)
                        regions.append(RoofRegion(
                            contour=pts, bbox=(x1, y1, bw, bh), area=area,
                            confidence=conf, source=f"yolo_{cls_name}",
                            color_score=color_score, texture_score=texture_score,
                            metadata={"class_id": cls_id},
                        ))

        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")

        return regions

    def _opencv_detect_regions(
        self, image: np.ndarray, is_real_photo: bool = True, **kwargs
    ) -> List[RoofRegion]:
        opencv_results = self._opencv_detector.detect(
            image, strict_mode=is_real_photo,
        )
        regions: List[RoofRegion] = []
        h, w = image.shape[:2]

        for r in opencv_results:
            verts = r.metadata.get("polygon_vertices", [])
            if len(verts) < 3:
                continue
            pts = np.array([[int(x), int(y)] for x, y in verts], dtype=np.int32)
            area = cv2.contourArea(pts)
            x, y, bw, bh = cv2.boundingRect(pts)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            roof_pixels = image[mask > 0]

            color_score = RoofColorClassifier.score_roof_color(roof_pixels)
            texture_score = RoofColorClassifier.score_roof_texture(
                image[y:y+bh, x:x+bw]
            )
            source = r.metadata.get("source", "opencv")
            contour_score = r.metadata.get("contour_score", 0.5)

            regions.append(RoofRegion(
                contour=pts, bbox=(x, y, bw, bh), area=area,
                confidence=contour_score, source=f"opencv_{source}",
                color_score=color_score, texture_score=texture_score,
                metadata=r.metadata,
            ))

        return regions

    def _merge_regions(
        self,
        regions: List[RoofRegion],
        iou_threshold: float = 0.5,
        min_roof_score: float = 0.25,
    ) -> List[RoofRegion]:
        if not regions:
            return []

        for r in regions:
            r.confidence = (
                0.40 * r.confidence +
                0.35 * r.color_score +
                0.25 * r.texture_score
            )

        regions = [r for r in regions if r.confidence >= min_roof_score]
        regions.sort(key=lambda r: r.confidence, reverse=True)

        kept: List[RoofRegion] = []
        for r in regions:
            x, y, bw, bh = r.bbox
            is_dup = False
            for k in kept:
                kx, ky, kbw, kbh = k.bbox
                ix = max(x, kx)
                iy = max(y, ky)
                iw = min(x + bw, kx + kbw) - ix
                ih = min(y + bh, ky + kbh) - iy
                if iw <= 0 or ih <= 0:
                    continue
                inter = iw * ih
                union = (bw * bh) + (kbw * kbh) - inter
                iou = inter / union if union > 0 else 0
                if iou > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(r)

        return kept

    def detect(
        self,
        image: np.ndarray,
        use_yolo: bool = True,
        use_opencv: bool = True,
        use_sahi: bool = False,
        yolo_conf: float = 0.2,
        min_roof_score: float = 0.25,
        max_results: int = 20,
        **kwargs,
    ) -> List[DetectionResult]:
        """
        Hybrid roof detection.

        Args:
            image: BGR numpy array
            use_yolo: Enable YOLO pass (uses building model if available)
            use_opencv: Enable OpenCV multi-strategy pass
            use_sahi: Enable SAHI tiling for large images
            yolo_conf: YOLO confidence threshold
            min_roof_score: Minimum combined roof-likeness score
            max_results: Max roof planes to return
        """
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge_density = np.mean(cv2.Canny(gray, 50, 150) > 0)
        is_real = edge_density > 0.05

        all_regions: List[RoofRegion] = []

        # YOLO pass — uses building model for real photos, COCO for others
        if use_yolo and self._yolo_loaded:
            yolo_regions = self._yolo_detect(
                image,
                conf_threshold=yolo_conf,
                use_sahi=use_sahi,
            )
            all_regions.extend(yolo_regions)
            logger.debug(f"YOLO ({self._model_type}): {len(yolo_regions)} regions")

        # OpenCV pass
        if use_opencv:
            opencv_regions = self._opencv_detect_regions(
                image, is_real_photo=is_real,
            )
            all_regions.extend(opencv_regions)
            logger.debug(f"OpenCV: {len(opencv_regions)} regions")

        # Merge with scoring
        merged = self._merge_regions(
            all_regions,
            min_roof_score=min_roof_score,
        )
        merged = merged[:max_results]

        # Convert to DetectionResult
        results: List[DetectionResult] = []
        for r in merged:
            bx, by, bw, bh = r.bbox
            verts = r.contour.reshape(-1, 2)
            polygon_vertices = [
                (float(px), float(py))
                for px, py in verts
                if 0 <= px < w and 0 <= py < h
            ]
            if len(polygon_vertices) < 3:
                continue

            geometry = BBoxGeometry(
                x_min=float(bx), y_min=float(by),
                x_max=float(bx + bw), y_max=float(by + bh),
            )
            cx = sum(v[0] for v in polygon_vertices) / len(polygon_vertices)
            cy = sum(v[1] for v in polygon_vertices) / len(polygon_vertices)

            results.append(DetectionResult(
                geometry=geometry,
                confidence=float(r.confidence),
                class_name="roof_plane",
                metadata={
                    "polygon_vertices": polygon_vertices,
                    "centroid": (cx, cy),
                    "source": r.source,
                    "area_pixels": r.area,
                    "color_score": round(r.color_score, 3),
                    "texture_score": round(r.texture_score, 3),
                    "combined_score": round(r.confidence, 3),
                },
            ))

        logger.info(
            f"HybridRoofDetector: {len(results)} roof planes "
            f"(model={self._model_type}, "
            f"{len(all_regions)} raw -> {len(merged)} merged)"
        )
        return results