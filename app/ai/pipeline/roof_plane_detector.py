"""
Roof plane detector pipeline component.
Detects multiple roof planes in an image using OpenCV contours and returns
rich DetectionResult objects with metadata including polygon_vertices,
centroid, confidence, source and area_pixels.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

from app.ai.ai_result import DetectionResult, BoundingBox
from app.core.logger import setup_logging

logger = setup_logging()

class RoofPlaneDetector:
    """Detects roof planes using classical CV heuristics and returns multiple DetectionResults."""

    def __init__(self):
        # Configuration defaults
        self.default_confidence = 0.9
        self.default_min_contour_area_ratio = 0.002
        self.default_max_contour_area_ratio = 0.98
        self.default_approx_factor = 0.02

    def _compute_centroid(self, pts: List[Tuple[int, int]]) -> Tuple[float, float]:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (float(sum(xs) / len(xs)), float(sum(ys) / len(ys)))

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image provided to RoofPlaneDetector.detect")

        h, w = image.shape[:2]
        image_area = float(h * w)

        # parameters
        blur_kernel = kwargs.get("blur_kernel_size", 5)
        morph_kernel_size = kwargs.get("morph_kernel_size", 5)
        approx_factor = kwargs.get("approx_poly_epsilon_factor", self.default_approx_factor)
        min_area_ratio = kwargs.get("min_contour_area_ratio", self.default_min_contour_area_ratio)
        max_area_ratio = kwargs.get("max_contour_area_ratio", self.default_max_contour_area_ratio)
        confidence = kwargs.get("confidence_score", self.default_confidence)

        # Preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        v = float(np.median(blurred))
        t1 = int(max(10, 0.66 * v))
        t2 = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, t1, t2)
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results: List[DetectionResult] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not (min_area_ratio * image_area <= area <= max_area_ratio * image_area):
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, approx_factor * perimeter, True)
            if len(approx) < 4:
                continue

            # compute bounding box and fill ratio
            x, y, rw, rh = cv2.boundingRect(approx)
            rect_area = float(rw * rh) if rw > 0 and rh > 0 else 0.0
            fill_ratio = (area / rect_area) if rect_area > 0 else 0.0
            if fill_ratio < 0.15:
                continue

            # extract polygon points
            try:
                pts = [ (int(p[0][0]), int(p[0][1])) for p in approx.reshape(-1,1,2) ]
            except Exception:
                pts = [ (int(p[0]), int(p[1])) for p in approx.reshape(-1,2) ]

            bbox = BoundingBox(x_min=float(x), y_min=float(y), x_max=float(x+rw), y_max=float(y+rh))
            centroid = self._compute_centroid(pts)

            metadata: Dict[str, Any] = {
                "source": "OpenCV_RoofPlaneDetector",
                "polygon_vertices": pts,
                "centroid": centroid,
                "confidence": float(confidence),
                "area_pixels": float(area)
            }

            det = DetectionResult(
                bounding_box=bbox,
                confidence=float(confidence),
                class_name="roof_plane",
                metadata=metadata
            )
            results.append(det)
            logger.info(f"RoofPlaneDetector: found plane with area {area:.1f} px and {len(pts)} vertices")

        # sort results by area descending
        results.sort(key=lambda r: r.metadata.get("area_pixels", 0.0), reverse=True)
        return results
