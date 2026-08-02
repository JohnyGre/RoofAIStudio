"""
Polygon Refiner: Converts masks to clean polygons with corner detection.

Pipeline:
  Mask → Contour Extraction → Douglas-Peucker Simplification → Corner Detection → Refined Polygon
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import cv2

from app.ai.config import get_detection_config
from app.core.logger import setup_logging

from .roof_plane import RoofPlane, RoofPlanePolygon

logger: logging.Logger = setup_logging()


class PolygonRefiner:
    """
    Refines masks into clean, meaningful polygons.
    
    This is the third stage of the scene understanding pipeline.
    
    Configuration from detection.yaml → geometry_refinement:
    - simplification_epsilon: Douglas-Peucker epsilon
    - corner_detection_method: "harris", "shi_tomasi", or "fast"
    - harris_block_size, harris_ksize, harris_k: Harris parameters
    - min_corner_quality: Minimum corner quality threshold
    - min_corner_distance: Minimum distance between corners
    - contour_epsilon_absolute: Absolute epsilon for simplification
    """
    
    def __init__(self):
        """Initialize the polygon refiner."""
        self.config = get_detection_config()
        self.refine_config = self.config.scene_understanding.geometry_refinement
        self.log_enabled = self.config.logging.log_geometry_refinement
        if self.log_enabled:
            logger.info("PolygonRefiner initialized")
    
    def refine_planes(self, planes: List[RoofPlane], image: Optional[np.ndarray] = None) -> List[RoofPlane]:
        """
        Refine all planes by converting masks to polygons.
        
        Args:
            planes: List of RoofPlane objects with masks
            image: Optional original image for corner detection refinement
        
        Returns:
            List of RoofPlane objects with computed polygons
        """
        refined_planes = []
        
        for plane in planes:
            try:
                if plane.mask is None or plane.mask.size == 0:
                    logger.warning(f"Plane {plane.id} has empty mask, skipping refinement")
                    refined_planes.append(plane)
                    continue
                
                # Extract and simplify contour
                polygon = self._mask_to_polygon(plane.mask)
                
                if polygon is None:
                    logger.warning(f"Failed to extract polygon from plane {plane.id}")
                    refined_planes.append(plane)
                    continue
                
                # Detect corners
                if image is not None:
                    polygon = self._detect_corners(plane.mask, polygon, image)
                
                plane.polygon = polygon
                refined_planes.append(plane)
                
            except Exception as e:
                logger.warning(f"Failed to refine plane {plane.id}: {e}")
                refined_planes.append(plane)
        
        if self.log_enabled:
            logger.info(f"Refined {len(refined_planes)} planes")
        
        return refined_planes
    
    def _mask_to_polygon(self, mask: np.ndarray) -> Optional[RoofPlanePolygon]:
        """
        Convert binary mask to simplified polygon.
        
        Steps:
        1. Extract contours from mask
        2. Find largest contour
        3. Apply Douglas-Peucker simplification
        4. Extract vertices
        
        Args:
            mask: Binary mask (H x W)
        
        Returns:
            RoofPlanePolygon or None if extraction fails
        """
        # Ensure mask is uint8
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return None
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            return None
        
        # Compute contour area and perimeter
        area = float(cv2.contourArea(largest_contour))
        perimeter = float(cv2.arcLength(largest_contour, closed=True))
        
        if area < 1.0 or perimeter < 1.0:
            return None
        
        # Douglas-Peucker simplification
        epsilon = max(
            self.refine_config.contour_epsilon_absolute,
            perimeter * self.refine_config.simplification_epsilon
        )
        simplified = cv2.approxPolyDP(largest_contour, epsilon, closed=True)
        
        # Extract vertices as list of tuples
        vertices = [(float(p[0][0]), float(p[0][1])) for p in simplified]
        
        if len(vertices) < 3:
            return None
        
        # Compute convex hull
        convex_hull = cv2.convexHull(largest_contour)
        convex_vertices = [(float(p[0][0]), float(p[0][1])) for p in convex_hull]
        
        polygon = RoofPlanePolygon(
            vertices=vertices,
            area=area,
            perimeter=perimeter,
            convex_hull=convex_vertices,
            corner_count=len(vertices),
            simplified=True,
        )
        
        return polygon
    
    def _detect_corners(
        self,
        mask: np.ndarray,
        polygon: RoofPlanePolygon,
        image: np.ndarray,
    ) -> RoofPlanePolygon:
        """
        Refine corners using Harris corner detection or other corner detectors.
        
        Args:
            mask: Binary mask
            polygon: Initial polygon from contour simplification
            image: Original image for corner detection
        
        Returns:
            Updated polygon with detected corners
        """
        if image is None or image.size == 0:
            return polygon
        
        try:
            # Convert to grayscale if needed
            if image.ndim == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Crop to region of interest (around mask)
            coords = np.argwhere(mask > 0)
            if len(coords) == 0:
                return polygon
            
            y_min, y_max = coords[:, 0].min(), coords[:, 0].max()
            x_min, x_max = coords[:, 1].min(), coords[:, 1].max()
            
            # Add padding
            pad = 10
            y_min = max(0, y_min - pad)
            y_max = min(gray.shape[0] - 1, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(gray.shape[1] - 1, x_max + pad)
            
            roi = gray[y_min:y_max+1, x_min:x_max+1].astype(np.float32)
            
            # Apply corner detector
            method = self.refine_config.corner_detection_method.lower()
            
            if method == "harris":
                corners = cv2.cornerHarris(
                    roi,
                    blockSize=self.refine_config.harris_block_size,
                    ksize=self.refine_config.harris_ksize,
                    k=self.refine_config.harris_k,
                )
            elif method == "shi_tomasi":
                corners = cv2.goodFeaturesToTrack(
                    roi,
                    maxCorners=100,
                    qualityLevel=self.refine_config.min_corner_quality,
                    minDistance=self.refine_config.min_corner_distance,
                )
                if corners is not None:
                    corners = corners.reshape(-1, 2)
                    # Convert back to Harris-like response
                    corners_response = np.zeros_like(roi)
                    for corner in corners:
                        cx, cy = int(corner[0]), int(corner[1])
                        if 0 <= cy < roi.shape[0] and 0 <= cx < roi.shape[1]:
                            corners_response[cy, cx] = 1.0
                    corners = corners_response
            else:
                logger.warning(f"Unknown corner detection method: {method}")
                return polygon
            
            # Extract corner locations
            if isinstance(corners, np.ndarray) and corners.dtype != np.float32:
                # Harris-like response map
                threshold = self.refine_config.min_corner_quality * corners.max()
                corner_coords = np.argwhere(corners > threshold)
            else:
                corner_coords = corners if corners is not None else np.array([])
            
            if len(corner_coords) > 0:
                # Convert back to original image coordinates
                corner_list = []
                for coord in corner_coords:
                    if len(coord) == 2:
                        y, x = coord
                        global_x = x + x_min
                        global_y = y + y_min
                        corner_list.append((float(global_x), float(global_y)))
                
                if corner_list:
                    polygon.corner_count = len(corner_list)
            
        except Exception as e:
            logger.debug(f"Corner detection failed: {e}")
        
        return polygon
