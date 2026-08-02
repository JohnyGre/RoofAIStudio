"""
Quality Scorer: Computes quality metrics for detected roof planes and groups.

Quality metrics:
- Mask quality: Smoothness and completeness of segmentation mask
- Polygon quality: How well the simplified polygon represents the mask
- Edge consistency: Consistency of edges (gradient-based)
- Convexity score: How close to convex shape (important for roof faces)
- Overall quality: Weighted combination of above metrics
"""

import logging
from typing import List
from dataclasses import dataclass
import numpy as np
import cv2

from app.ai.config import get_detection_config
from app.core.logger import setup_logging

from .roof_plane import RoofPlane
from .roof_group import RoofGroup

logger: logging.Logger = setup_logging()


@dataclass
class QualityMetrics:
    """Container for all quality metrics of a plane."""
    mask_quality: float = 0.0      # Smoothness and connectivity
    polygon_quality: float = 0.0   # How well polygon represents mask
    edge_consistency: float = 0.0  # Edge gradient consistency
    convexity_score: float = 0.0   # Convexity measure (0.0 = perfect, 1.0 = very concave)
    overall_score: float = 0.0     # Weighted combination
    quality_level: str = "unknown" # excellent, good, fair, poor


class QualityScorer:
    """
    Scores the quality of detected roof planes.
    
    This stage runs after geometry refinement and provides quality estimates
    that determine whether SAM refinement might be beneficial.
    
    Configuration from detection.yaml → quality_scoring:
    - mask_quality_weight, polygon_quality_weight, etc.: Quality component weights
    - min_solidity, min_extent: Shape quality thresholds
    - edge_consistency_threshold: Edge quality threshold
    - max_concavity_ratio: Maximum allowed concavity
    - quality_*_threshold: Quality level thresholds
    """
    
    def __init__(self):
        """Initialize the quality scorer."""
        self.config = get_detection_config()
        self.quality_config = self.config.scene_understanding.quality_scoring
        self.log_enabled = self.config.logging.log_quality_scoring
        if self.log_enabled:
            logger.info("QualityScorer initialized")
    
    def score_planes(self, planes: List[RoofPlane]) -> List[RoofPlane]:
        """
        Compute quality scores for all planes.
        
        Args:
            planes: List of RoofPlane objects with masks
        
        Returns:
            Planes with quality_score populated
        """
        for plane in planes:
            try:
                metrics = self.compute_metrics(plane)
                plane.quality_score = metrics.overall_score
                plane.solidity = metrics.polygon_quality
                plane.extent = metrics.mask_quality
                plane.edge_consistency = metrics.edge_consistency
                plane.convexity_ratio = metrics.convexity_score
            except Exception as e:
                logger.warning(f"Failed to score plane {plane.id}: {e}")
                plane.quality_score = 0.5  # Default score
        
        if self.log_enabled:
            avg_score = np.mean([p.quality_score for p in planes])
            logger.info(f"Scored {len(planes)} planes (avg quality: {avg_score:.2f})")
        
        return planes
    
    def score_groups(self, groups: List[RoofGroup]) -> List[RoofGroup]:
        """
        Compute quality scores for all groups (averaged from plane scores).
        
        Args:
            groups: List of RoofGroup objects
        
        Returns:
            Groups with group_quality_score populated
        """
        for group in groups:
            if group.planes:
                group.group_quality_score = np.mean([p.quality_score for p in group.planes])
            else:
                group.group_quality_score = 0.0
        
        if self.log_enabled:
            avg_group_score = np.mean([g.group_quality_score for g in groups]) if groups else 0.0
            logger.info(f"Scored {len(groups)} groups (avg quality: {avg_group_score:.2f})")
        
        return groups
    
    def compute_metrics(self, plane: RoofPlane) -> QualityMetrics:
        """
        Compute all quality metrics for a single plane.
        
        Args:
            plane: RoofPlane object
        
        Returns:
            QualityMetrics object
        """
        metrics = QualityMetrics()
        
        # 1. Mask Quality (smoothness, connectivity)
        if plane.mask is not None and plane.mask.size > 0:
            metrics.mask_quality = self._compute_mask_quality(plane.mask)
        
        # 2. Polygon Quality (solidity and extent)
        if plane.polygon is not None:
            metrics.polygon_quality = self._compute_polygon_quality(plane)
        else:
            metrics.polygon_quality = 0.5  # Default if no polygon
        
        # 3. Edge Consistency
        if plane.mask is not None and plane.mask.size > 0:
            metrics.edge_consistency = self._compute_edge_consistency(plane.mask)
        
        # 4. Convexity Score
        if plane.polygon is not None and plane.polygon.convex_hull is not None:
            metrics.convexity_score = self._compute_convexity_score(plane)
        
        # 5. Overall Quality Score (weighted combination)
        metrics.overall_score = (
            metrics.mask_quality * self.quality_config.mask_quality_weight +
            metrics.polygon_quality * self.quality_config.polygon_quality_weight +
            metrics.edge_consistency * self.quality_config.edge_consistency_weight +
            (1.0 - metrics.convexity_score) * self.quality_config.convexity_score_weight
        )
        
        # Clamp to [0, 1]
        metrics.overall_score = np.clip(metrics.overall_score, 0.0, 1.0)
        
        # Determine quality level
        metrics.quality_level = self._classify_quality(metrics.overall_score)
        
        return metrics
    
    def _compute_mask_quality(self, mask: np.ndarray) -> float:
        """
        Compute mask quality based on smoothness and connectivity.
        
        Smooth masks with clean boundaries score higher.
        
        Args:
            mask: Binary mask
        
        Returns:
            Quality score (0.0-1.0)
        """
        # Compute boundary length
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, closed=True)
        
        # Smoothness metric: area / perimeter^2
        # Circle has best score, elongated shapes score lower
        if perimeter > 0:
            compactness = (4 * np.pi * area) / (perimeter ** 2)
            compactness = np.clip(compactness, 0.0, 1.0)
        else:
            compactness = 0.0
        
        return float(compactness)
    
    def _compute_polygon_quality(self, plane: RoofPlane) -> float:
        """
        Compute polygon quality based on solidity and extent.
        
        Args:
            plane: RoofPlane with polygon
        
        Returns:
            Quality score (0.0-1.0)
        """
        if plane.mask is None or plane.polygon is None:
            return 0.5
        
        # Solidity: area of polygon / area of convex hull
        mask_uint8 = (plane.mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.5
        
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if plane.polygon.convex_hull:
            hull_pts = np.array(plane.polygon.convex_hull, dtype=np.float32)
            if len(hull_pts) >= 3:
                hull_area = cv2.contourArea(hull_pts)
                if hull_area > 0:
                    solidity = area / hull_area
                    solidity = np.clip(solidity, 0.0, 1.0)
                else:
                    solidity = 0.5
            else:
                solidity = 0.5
        else:
            solidity = 0.5
        
        # Extent: area of polygon / area of bounding box
        bbox = plane.bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if bbox_area > 0:
            extent = area / bbox_area
            extent = np.clip(extent, 0.0, 1.0)
        else:
            extent = 0.5
        
        # Combine solidity and extent
        quality = (solidity + extent) / 2.0
        
        return float(quality)
    
    def _compute_edge_consistency(self, mask: np.ndarray) -> float:
        """
        Compute edge consistency based on gradient magnitude along boundaries.
        
        Masks with consistent, strong edges score higher.
        
        Args:
            mask: Binary mask
        
        Returns:
            Edge consistency score (0.0-1.0)
        """
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        
        # Compute gradients
        gx = cv2.Sobel(mask_uint8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(mask_uint8, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        
        # Find edge pixels (high gradient)
        edges = magnitude > np.percentile(magnitude, 75)
        
        if np.sum(edges) == 0:
            return 0.5
        
        # Compute consistency as inverse of variance in gradient magnitude
        edge_magnitudes = magnitude[edges]
        if len(edge_magnitudes) > 0:
            # Normalize to [0, 1]
            normalized = (edge_magnitudes - edge_magnitudes.min()) / (edge_magnitudes.max() - edge_magnitudes.min() + 1e-6)
            # High variance means inconsistent edges
            variance = np.var(normalized)
            consistency = 1.0 - np.clip(variance, 0.0, 1.0)
        else:
            consistency = 0.5
        
        return float(consistency)
    
    def _compute_convexity_score(self, plane: RoofPlane) -> float:
        """
        Compute convexity score (measure of concavity).
        
        Perfect convex shapes score 0.0, highly concave score closer to 1.0.
        
        Args:
            plane: RoofPlane with polygon
        
        Returns:
            Convexity score (0.0 = convex, 1.0 = highly concave)
        """
        if plane.mask is None or plane.polygon is None or not plane.polygon.vertices:
            return 0.5
        
        mask_uint8 = (plane.mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.5
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute convexity defects
        if len(largest_contour) >= 4:
            hull = cv2.convexHull(largest_contour, returnPoints=False)
            defects = cv2.convexityDefects(largest_contour, hull)
            
            if defects is not None and len(defects) > 0:
                # Average convexity defect depth
                defect_depths = defects[:, 0, 3]
                max_defect = float(defect_depths.max())
                
                # Normalize by hull perimeter
                hull_pts = largest_contour[hull.flatten()]
                hull_perimeter = cv2.arcLength(hull_pts, closed=True)
                
                if hull_perimeter > 0:
                    normalized_defect = max_defect / hull_perimeter
                    concavity = np.clip(normalized_defect, 0.0, 1.0)
                else:
                    concavity = 0.0
            else:
                concavity = 0.0
        else:
            concavity = 0.0
        
        return float(concavity)
    
    def _classify_quality(self, score: float) -> str:
        """Classify quality score into levels."""
        if score >= self.quality_config.quality_excellent_threshold:
            return "excellent"
        elif score >= self.quality_config.quality_good_threshold:
            return "good"
        elif score >= self.quality_config.quality_fair_threshold:
            return "fair"
        else:
            return "poor"
