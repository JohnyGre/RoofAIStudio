"""
Roof plane detector pipeline component.
Detects multiple roof planes in an image using OpenCV contours and returns
rich DetectionResult objects with metadata including polygon_vertices,
centroid, confidence, source and area_pixels.

Improvements in this version:
- Combine adaptive thresholding with Canny edges to reduce stray contours.
- Use convex hull + smaller approx epsilon for tighter polygon approximation.
- Filter contours by solidity and fill ratio to avoid polygons outside the roof.
- Add detailed debug logging for rejected contours to help tuning.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

from app.ai.ai_result import DetectionResult, BoundingBox, PolygonGeometry
from app.ai.pipeline.base_detector import BaseSubDetector
from app.core.logger import setup_logging

logger = setup_logging()

class RoofPlaneDetector(BaseSubDetector):
    """Detects roof planes using classical CV heuristics and returns multiple DetectionResults."""

    def __init__(self):
        # Configuration defaults
        self.default_confidence = 0.9
        self.default_min_contour_area_ratio = 0.001  # allow smaller planes
        self.default_max_contour_area_ratio = 0.98
        self.default_approx_factor = 0.01  # tighter approximation
        self.default_min_fill_ratio = 0.20  # require more filled bbox (relaxed)
        self.default_min_solidity = 0.45

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
        morph_kernel_size = kwargs.get("morph_kernel_size", 3)
        approx_factor = kwargs.get("approx_poly_epsilon_factor", self.default_approx_factor)
        min_area_ratio = kwargs.get("min_contour_area_ratio", self.default_min_contour_area_ratio)
        max_area_ratio = kwargs.get("max_contour_area_ratio", self.default_max_contour_area_ratio)
        min_fill_ratio = kwargs.get("min_fill_ratio", self.default_min_fill_ratio)
        min_solidity = kwargs.get("min_solidity", self.default_min_solidity)
        confidence = kwargs.get("confidence_score", self.default_confidence)

        # --- Preprocessing ---
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Slight bilateral filter to preserve edges while reducing noise
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Adaptive threshold (helps pick up roof regions under varying illumination)
        try:
            th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        except Exception:
            # fallback to Otsu
            _, th = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If threshold image has mostly white pixels, invert it so foreground objects are white
        mean_th = float(np.mean(th))
        if mean_th > 127:
            logger.debug(f"Inverting threshold mask (mean={mean_th:.1f})")
            th = cv2.bitwise_not(th)

        # Canny edges on a lightly blurred image
        blurred = cv2.GaussianBlur(denoised, (blur_kernel, blur_kernel), 0)
        v = float(np.median(blurred))
        t1 = int(max(10, 0.66 * v))
        t2 = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, t1, t2)

        # Combine threshold mask and edges to get stronger contours
        combined = cv2.bitwise_or(th, edges)

        # Morphological closing + opening to remove small holes and noise
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

        # Final dilate to make contours continuous
        morph = cv2.dilate(morph, kernel, iterations=1)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"RoofPlaneDetector: found {len(contours)} raw contours after morphology")

        results: List[DetectionResult] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            logger.info(f"Raw contour area: {area:.1f} px")
            if not (min_area_ratio * image_area <= area <= max_area_ratio * image_area):
                logger.debug(f"Skipping contour by area: {area:.1f} px")
                continue

            # compute convex hull and solidity
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0
            solidity = (area / hull_area) if hull_area > 0 else 0.0

            x, y, rw, rh = cv2.boundingRect(contour)
            rect_area = float(rw * rh) if rw > 0 and rh > 0 else 0.0
            fill_ratio = (area / rect_area) if rect_area > 0 else 0.0

            logger.info(f"Contour stats: area={area:.1f}, rect_area={rect_area:.1f}, fill_ratio={fill_ratio:.3f}, solidity={solidity:.3f}")

            if fill_ratio < min_fill_ratio:
                logger.debug(f"Skipping contour by fill_ratio: {fill_ratio:.2f} (area={area:.1f})")
                continue
            if solidity < min_solidity:
                logger.debug(f"Skipping contour by solidity: {solidity:.2f} (area={area:.1f})")
                continue

            perimeter = cv2.arcLength(contour, True)
            # Approximate contour using perimeter-based epsilon
            approx = cv2.approxPolyDP(contour, approx_factor * perimeter, True)
            if len(approx) < 3:
                logger.debug(f"Skipping small approx vertices: {len(approx)}")
                continue

            # Optionally smooth polygon by approximating hull then approximating hull
            try:
                hull_pts = hull.reshape(-1, 2)
                # approximate hull for simpler polygon
                hull_perim = cv2.arcLength(hull, True)
                hull_approx = cv2.approxPolyDP(hull, max(1.0, 0.5 * approx_factor * hull_perim), True)
                pts = [(int(p[0][0]), int(p[0][1])) for p in hull_approx.reshape(-1,1,2)]
            except Exception:
                # fallback to contour approx
                pts = [(int(p[0][0]), int(p[0][1])) for p in approx.reshape(-1,1,2)]

            # Ensure polygon is within image bounds and unique
            valid_pts = []
            for (px, py) in pts:
                if 0 <= px < w and 0 <= py < h:
                    valid_pts.append((px, py))
            if len(valid_pts) < 3:
                logger.debug("Skipping contour: insufficient valid polygon points after bounds check")
                continue

            bbox = BoundingBox(x_min=float(x), y_min=float(y), x_max=float(x+rw), y_max=float(y+rh))
            centroid = self._compute_centroid(valid_pts)

            metadata: Dict[str, Any] = {
                "source": "OpenCV_RoofPlaneDetector",
                "polygon_vertices": valid_pts,
                "centroid": centroid,
                "confidence": float(confidence),
                "area_pixels": float(area),
                "fill_ratio": float(fill_ratio),
                "solidity": float(solidity)
            }

            det = DetectionResult(
                class_name="roof_plane",
                geometry=PolygonGeometry(vertices=[(float(px), float(py)) for px, py in valid_pts]),
                confidence=float(confidence),
                metadata=metadata
            )
            results.append(det)
            logger.info(f"RoofPlaneDetector: found plane with area {area:.1f} px, vertices={len(valid_pts)}, fill_ratio={fill_ratio:.2f}, solidity={solidity:.2f}")

        # sort results by area descending
        results.sort(key=lambda r: r.metadata.get("area_pixels", 0.0), reverse=True)
        return results
