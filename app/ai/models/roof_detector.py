"""
This module implements a basic OpenCV-based roof detection.
"""

import uuid
from typing import Any, Dict, List, Union, Optional
import logging

import cv2
import numpy as np

from app.ai.models.vision_detector import VisionDetector
from app.ai.ai_result import DetectionResult, BoundingBox, SegmentationResult, GeometryPredictionResult
from app.core.logger import setup_logging

logger = setup_logging()

class RoofDetector(VisionDetector):
    """
    A basic roof detector using OpenCV operations to find the largest roof-like contour.
    This serves as a functional placeholder until more advanced ML models are integrated.
    """
    MODEL_NAME = "OpenCV_Roof_Detector"
    VERSION = "0.1.0-opencv"

    def __init__(self, model_id: Optional[uuid.UUID] = None):
        super().__init__(self.MODEL_NAME, self.VERSION, model_id)
        self._model_info: Dict[str, Any] = {
            "name": self.MODEL_NAME,
            "version": self.VERSION,
            "description": "Basic roof detector using OpenCV contour finding, morphology, and polygon approximation.",
            "input_requirements": "BGR NumPy array image",
            "output_format": "List[DetectionResult]",
            "trained_classes": ["roof_area"] # This detector primarily finds general roof areas
        }

    def load(self, model_path: str = "", device: str = "cpu", **kwargs) -> None:
        """
        Loads the detector. No actual model file is loaded for this OpenCV-based detector.
        """
        logger.info(f"OpenCV RoofDetector '{self.model_name}' loaded (no external model file).")
        self._is_loaded = True

    def detect(self, image: np.ndarray, **kwargs) -> List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
        """
        Detects roof areas in the input image using OpenCV contour processing.

        Args:
            image (np.ndarray): The input image data (NumPy array, BGR format).
            **kwargs: Additional parameters for OpenCV processing:
                      - `blur_kernel_size` (int): Kernel size for Gaussian blur (default 5).
                      - `canny_threshold1` (int): First threshold for the Canny edge detector (default 50).
                      - `canny_threshold2` (int): Second threshold for the Canny edge detector (default 150).
                      - `morph_kernel_size` (int): Kernel size for morphological operations (default 5).
                      - `min_contour_area_ratio` (float): Minimum contour area as a ratio of total image area (default 0.01).
                      - `max_contour_area_ratio` (float): Maximum contour area as a ratio of total image area (default 0.95).
                      - `approx_poly_epsilon_factor` (float): Factor for cv2.approxPolyDP (default 0.02).
                      - `confidence_score` (float): Fixed confidence score for detected roofs (default 0.9).

        Returns:
            List[DetectionResult]: A list containing a single DetectionResult for the largest roof-like contour found.
                                   Returns an empty list if no suitable contour is found.
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded. Call load() first.")
        if not self.validate(image):
            raise ValueError("Invalid image input for detection.")

        height, width, _ = image.shape
        image_area = float(height * width)

        # --- Parameters from kwargs or defaults ---
        blur_kernel_size = kwargs.get("blur_kernel_size", 5)
        # Use automatic Canny thresholds by default for robustness
        canny_threshold1 = kwargs.get("canny_threshold1", None)
        canny_threshold2 = kwargs.get("canny_threshold2", None)
        morph_kernel_size = kwargs.get("morph_kernel_size", 5)
        # Lower default to 0.2% of image area to detect smaller roofs in screenshots
        min_contour_area_ratio = kwargs.get("min_contour_area_ratio", 0.002)
        max_contour_area_ratio = kwargs.get("max_contour_area_ratio", 0.98)
        approx_poly_epsilon_factor = kwargs.get("approx_poly_epsilon_factor", 0.02)
        confidence_score = kwargs.get("confidence_score", 0.9)

        # 1. Grayscale and Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

        # 2. Edge Detection (use median-based automatic thresholds if not provided)
        if canny_threshold1 is None or canny_threshold2 is None:
            v = float(np.median(blurred))
            canny_threshold1 = int(max(10, 0.66 * v))
            canny_threshold2 = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

        # 3. Morphological Operations to close gaps in edges (closing + dilation)
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        closed_edges = cv2.dilate(closed, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_roof_contour = None
        max_contour_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area ratio
            if not (min_contour_area_ratio * image_area <= area <= max_contour_area_ratio * image_area):
                continue

            # Approximate polygon to simplify contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, approx_poly_epsilon_factor * perimeter, True)

            # We are looking for a closed shape, ideally with more than 3 vertices
            if len(approx) >= 4: # A roof plane is typically a quadrilateral or more complex polygon
                # Compute bounding rectangle and fill ratio to exclude very thin or spurious contours
                rx, ry, rw, rh = cv2.boundingRect(approx)
                rect_area = float(rw * rh) if (rw > 0 and rh > 0) else 0.0
                fill_ratio = (area / rect_area) if rect_area > 0 else 0.0
                # Heuristics: at least 20% fill of bounding rect and larger area than previous
                if fill_ratio >= 0.20 and area > max_contour_area:
                    max_contour_area = area
                    largest_roof_contour = approx

        results: List[DetectionResult] = []
        if largest_roof_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_roof_contour)
            bbox = BoundingBox(x_min=float(x), y_min=float(y), x_max=float(x + w), y_max=float(y + h))
            
            # Extract polygon vertices from the approx contour (ensure Nx2 list of ints)
            try:
                poly_pts = [ (int(p[0]), int(p[1])) for p in largest_roof_contour.reshape(-1, 2) ]
            except Exception:
                poly_pts = [ (int(p[0]), int(p[1])) for p in largest_roof_contour ]

            metadata = {"source": "OpenCV Roof Detection", "contour_area": max_contour_area, "contour_polygon": poly_pts}

            results.append(DetectionResult(
                bounding_box=bbox,
                confidence=confidence_score,
                class_name="roof_area",
                metadata=metadata
            ))
            logger.info(f"Detected largest roof area with bounding box: {bbox} and polygon with {len(poly_pts)} vertices")
        else:
            logger.info("No suitable roof-like contour found.")
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns detailed information about this detector.
        """
        return self._model_info

    def validate(self, data: Any) -> bool:
        """
        Validates the input data for the detector.
        Ensures it's a NumPy array with at least 2 dimensions.
        """
        return isinstance(data, np.ndarray) and data.ndim >= 2 and data.shape[0] > 0 and data.shape[1] > 0
