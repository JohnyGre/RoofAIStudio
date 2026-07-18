"""
This module defines the RoofPlaneDetector, responsible for detecting roof planes
from an image using OpenCV.
"""

import uuid
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional

from app.ai.ai_result import DetectionResult, BoundingBox
from app.core.logger import setup_logging
from app.geometry.point import Point2D
from app.geometry.polygon import Polygon2D

logger = setup_logging()

class RoofPlaneDetector:
    """
    Detects individual roof planes in an image using OpenCV contour processing.

    Note:
        Final polygon vertices come from approxPolyDP on the original
        contour, NOT from the convex hull, to preserve concave roof
        features (valleys, L/T-shaped footprints). Convex hull is used
        only for the solidity rejection filter.
    """

    def __init__(self):
        logger.info("RoofPlaneDetector initialized.")

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """
        Detects roof planes in the input image using OpenCV contour processing.

        Args:
            image (np.ndarray): The input image data (NumPy array, BGR format).
            **kwargs: Additional parameters for OpenCV processing:
                      - `blur_kernel_size` (int): Kernel size for Gaussian blur (default 5).
                      - `canny_threshold1` (int): First threshold for the Canny edge detector (default 50).
                      - `canny_threshold2` (int): Second threshold for the Canny edge detector (default 150).
                      - `morph_kernel_size` (int): Kernel size for morphological operations (default 5).
                      - `min_contour_area_ratio` (float): Minimum contour area as a ratio of total image area (default 0.005).
                      - `max_contour_area_ratio` (float): Maximum contour area as a ratio of total image area (default 0.95).
                      - `approx_poly_epsilon_factor` (float): Factor for cv2.approxPolyDP (default 0.02).
                      - `min_vertices` (int): Minimum number of vertices for an approximated polygon (default 4).
                      - `confidence_score` (float): Fixed confidence score for detected roof planes (default 0.9).

        Returns:
            List[DetectionResult]: A list of DetectionResult objects for each detected roof plane.
        """
        results: List[DetectionResult] = []
        height, width, _ = image.shape
        image_area = float(height * width)

        # --- Parameters from kwargs or defaults ---
        blur_kernel_size = kwargs.get("blur_kernel_size", 5)
        canny_threshold1 = kwargs.get("canny_threshold1", 30) # Znížené prahy
        canny_threshold2 = kwargs.get("canny_threshold2", 90) # Znížené prahy
        morph_kernel_size = kwargs.get("morph_kernel_size", 3) # Zmenená veľkosť jadra
        min_contour_area_ratio = kwargs.get("min_contour_area_ratio", 0.001) # Znížený prah
        max_contour_area_ratio = kwargs.get("max_contour_area_ratio", 0.99) # Zvýšený prah
        approx_poly_epsilon_factor = kwargs.get("approx_poly_epsilon_factor", 0.04) # Zvýšená tolerancia
        min_vertices = kwargs.get("min_vertices", 3) # Znížený počet vrcholov (min. 3 pre polygón)
        confidence_score = kwargs.get("confidence_score", 0.9)

        # 1. Grayscale and Blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

        # 2. Edge Detection
        edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

        # 3. Morphological Operations to close gaps in edges and enhance contours
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        # Closing operation (dilation followed by erosion) to close small holes and gaps
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        # Opening operation (erosion followed by dilation) to remove small noise
        opened_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Find Contours
        contours, _ = cv2.findContours(opened_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area ratio
            if not (min_contour_area_ratio * image_area <= area <= max_contour_area_ratio * image_area):
                continue

            # Approximate polygon to simplify contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, approx_poly_epsilon_factor * perimeter, True)

            # Filter by number of vertices
            if len(approx) >= min_vertices: # Odstránená podmienka cv2.isContourConvex(approx)
                x, y, w, h = cv2.boundingRect(approx)
                bbox = BoundingBox(x_min=float(x), y_min=float(y), x_max=float(x + w), y_max=float(y + h))
                
                # Extract polygon vertices as a list of (x, y) tuples
                polygon_vertices_list = [(float(p[0][0]), float(p[0][1])) for p in approx]
                
                # Calculate centroid (simple average for now, could use shapely for accuracy)
                centroid_x = sum(p[0] for p in polygon_vertices_list) / len(polygon_vertices_list)
                centroid_y = sum(p[1] for p in polygon_vertices_list) / len(polygon_vertices_list)

                results.append(DetectionResult(
                    bounding_box=bbox,
                    confidence=confidence_score,
                    class_name="roof_plane",
                    metadata={
                        "polygon_vertices": polygon_vertices_list,
                        "centroid": (centroid_x, centroid_y),
                        "source": "opencv",
                        "area_pixels": area
                    }
                ))
        
        logger.info(f"RoofPlaneDetector completed. Found {len(results)} roof planes.")
        return results
