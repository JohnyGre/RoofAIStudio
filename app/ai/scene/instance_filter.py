"""
Instance Filter: Filters YOLO detections based on quality criteria.

Removes detections that don't meet minimum quality standards:
- Minimum area
- Confidence threshold
- Excessive overlap (duplicates)
- Masks outside image boundaries
- Masks touching image borders
"""

import logging
from typing import List, Tuple
import numpy as np
import cv2

from app.ai.config import get_detection_config
from app.core.logger import setup_logging

from .roof_plane import RoofPlane

logger: logging.Logger = setup_logging()


class InstanceFilter:
    """
    Filters raw YOLO detection masks based on quality criteria.
    
    This is the first stage of the scene understanding pipeline.
    
    Configuration from detection.yaml → instance_filtering:
    - min_mask_area: Minimum mask area in pixels
    - confidence_threshold: Minimum confidence
    - duplicate_iou_threshold: IOU threshold for marking as duplicates
    - min_border_distance: Minimum distance from image border
    - discard_touching_border: Whether to discard masks touching border
    - max_mask_area: Maximum area (for filtering oversized noise)
    """
    
    def __init__(self):
        """Initialize the instance filter."""
        self.config = get_detection_config()
        self.instance_config = self.config.scene_understanding.instance_filtering
        self.log_enabled = self.config.logging.log_instance_filtering
        if self.log_enabled:
            logger.info("InstanceFilter initialized")
    
    def filter_detections(
        self,
        masks: List[np.ndarray],
        confidences: List[float],
        image_size: Tuple[int, int],
    ) -> List[RoofPlane]:
        """
        Filter detections and create RoofPlane objects.
        
        Args:
            masks: List of binary segmentation masks (H x W)
            confidences: List of confidence scores (0.0-1.0)
            image_size: Image size (height, width)
        
        Returns:
            List of filtered RoofPlane objects
        """
        if len(masks) != len(confidences):
            raise ValueError("Number of masks must match number of confidences")
        
        height, width = image_size
        
        # Step 1: Filter by confidence and area
        candidates = []
        for mask, conf in zip(masks, confidences):
            if conf < self.instance_config.confidence_threshold:
                if self.log_enabled:
                    logger.debug(f"Filtering out mask with low confidence: {conf:.2f}")
                continue
            
            area = np.sum(mask > 0)
            if area < self.instance_config.min_mask_area:
                if self.log_enabled:
                    logger.debug(f"Filtering out mask with area too small: {area} pixels")
                continue
            
            if self.instance_config.max_mask_area > 0 and area > self.instance_config.max_mask_area:
                if self.log_enabled:
                    logger.debug(f"Filtering out mask with area too large: {area} pixels")
                continue
            
            candidates.append((mask, conf, area))
        
        if self.log_enabled:
            logger.debug(f"Confidence/area filtering: {len(masks)} → {len(candidates)} masks")
        
        # Step 2: Filter by border proximity
        if self.instance_config.discard_touching_border or self.instance_config.min_border_distance > 0:
            candidates = self._filter_border_masks(candidates, image_size)
        
        # Step 3: Remove duplicates (high IOU overlap)
        candidates = self._remove_duplicates(candidates)
        
        # Step 4: Create RoofPlane objects with bounding boxes
        planes = []
        for mask, conf, area in candidates:
            try:
                plane = self._mask_to_roof_plane(mask, conf, image_size)
                planes.append(plane)
            except Exception as e:
                logger.warning(f"Failed to convert mask to RoofPlane: {e}")
        
        if self.log_enabled:
            logger.info(f"Instance filtering complete: {len(masks)} → {len(planes)} planes")
        
        return planes
    
    def _filter_border_masks(
        self,
        candidates: List[Tuple[np.ndarray, float, float]],
        image_size: Tuple[int, int],
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Filter masks that touch or are too close to image borders.
        
        Args:
            candidates: List of (mask, confidence, area) tuples
            image_size: Image size (height, width)
        
        Returns:
            Filtered list
        """
        height, width = image_size
        min_dist = self.instance_config.min_border_distance
        discard_touching = self.instance_config.discard_touching_border
        
        filtered = []
        for mask, conf, area in candidates:
            # Find non-zero coordinates
            coords = np.argwhere(mask > 0)
            if len(coords) == 0:
                continue
            
            # Get min/max coordinates
            y_coords = coords[:, 0]
            x_coords = coords[:, 1]
            min_y, max_y = y_coords.min(), y_coords.max()
            min_x, max_x = x_coords.min(), x_coords.max()
            
            # Check distance to borders
            if discard_touching:
                if min_x == 0 or max_x == width - 1 or min_y == 0 or max_y == height - 1:
                    if self.log_enabled:
                        logger.debug(f"Filtering out mask touching image border")
                    continue
            
            if min_dist > 0:
                if min_x < min_dist or max_x > width - 1 - min_dist:
                    if self.log_enabled:
                        logger.debug(f"Filtering out mask too close to left/right border")
                    continue
                if min_y < min_dist or max_y > height - 1 - min_dist:
                    if self.log_enabled:
                        logger.debug(f"Filtering out mask too close to top/bottom border")
                    continue
            
            filtered.append((mask, conf, area))
        
        return filtered
    
    def _remove_duplicates(
        self,
        candidates: List[Tuple[np.ndarray, float, float]],
    ) -> List[Tuple[np.ndarray, float, float]]:
        """
        Remove duplicate detections (high IOU overlap).
        
        Args:
            candidates: List of (mask, confidence, area) tuples
        
        Returns:
            Filtered list with duplicates removed
        """
        iou_threshold = self.instance_config.duplicate_iou_threshold
        
        # Sort by confidence descending (keep highest confidence duplicates)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        kept = []
        for mask1, conf1, area1 in candidates:
            is_duplicate = False
            for mask2, conf2, area2 in kept:
                iou = self._compute_iou(mask1, mask2)
                if iou > iou_threshold:
                    is_duplicate = True
                    if self.log_enabled:
                        logger.debug(f"Removing duplicate mask (IOU={iou:.2f})")
                    break
            
            if not is_duplicate:
                kept.append((mask1, conf1, area1))
        
        return kept
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IOU) between two masks.
        
        Args:
            mask1, mask2: Binary masks
        
        Returns:
            IOU score (0.0-1.0)
        """
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _mask_to_roof_plane(
        self,
        mask: np.ndarray,
        confidence: float,
        image_size: Tuple[int, int],
    ) -> RoofPlane:
        """
        Convert mask and confidence to RoofPlane object.
        
        Computes bounding box, average color, and area.
        
        Args:
            mask: Binary mask
            confidence: Confidence score
            image_size: Image size (height, width)
        
        Returns:
            RoofPlane object
        """
        # Compute bounding box
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            raise ValueError("Empty mask (no positive pixels)")
        
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        bbox = (float(x_min), float(y_min), float(x_max), float(y_max))
        
        # Placeholder for color (will be computed with actual image)
        color_rgb = (128, 128, 128)
        
        # Compute area
        area_pixels = np.sum(mask > 0)
        
        plane = RoofPlane(
            mask=mask,
            confidence=confidence,
            bbox=bbox,
            color_rgb=color_rgb,
            area_pixels=float(area_pixels),
        )
        
        return plane
