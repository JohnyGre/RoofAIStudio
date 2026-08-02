"""
Roof Grouper: Groups roof planes into buildings based on spatial proximity.

Multiple roof planes belonging to the same building are grouped together,
reducing the output complexity and enabling building-level analysis.
"""

import logging
from typing import List, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from app.ai.config import get_detection_config
from app.core.logger import setup_logging

from .roof_plane import RoofPlane
from .roof_group import RoofGroup

logger: logging.Logger = setup_logging()


class RoofGrouper:
    """
    Groups roof planes into buildings based on spatial proximity.
    
    This is the second stage of the scene understanding pipeline.
    
    Configuration from detection.yaml → roof_grouping:
    - grouping_distance_threshold: Distance threshold for grouping
    - spatial_clustering_max_distance: Max distance for K-means clustering
    - min_planes_per_group: Minimum planes to form a valid group
    - confidence_weighting_mode: How to compute group confidence
    """
    
    def __init__(self):
        """Initialize the roof grouper."""
        self.config = get_detection_config()
        self.grouping_config = self.config.scene_understanding.roof_grouping
        self.log_enabled = self.config.logging.log_roof_grouping
        if self.log_enabled:
            logger.info("RoofGrouper initialized")
    
    def group_planes(self, planes: List[RoofPlane]) -> List[RoofGroup]:
        """
        Group planes into buildings using spatial clustering.
        
        Args:
            planes: List of RoofPlane objects from instance filtering
        
        Returns:
            List of RoofGroup objects
        """
        if not planes:
            return []
        
        # Extract centroids from plane bounding boxes
        centroids = np.array([p.centroid for p in planes])
        
        if len(planes) == 1:
            # Single plane - create single group
            group = RoofGroup(planes=planes.copy())
            for plane in group.planes:
                plane.group_id = group.id
            if self.log_enabled:
                logger.info("Grouping: 1 plane → 1 group")
            return [group]
        
        # Use DBSCAN for spatial clustering
        distance_threshold = self.grouping_config.grouping_distance_threshold
        
        try:
            # DBSCAN requires epsilon and min_samples
            clustering = DBSCAN(
                eps=distance_threshold,
                min_samples=1,  # Allow singleton clusters
            ).fit(centroids)
            
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}. Using single group.")
            group = RoofGroup(planes=planes.copy())
            for plane in group.planes:
                plane.group_id = group.id
            return [group]
        
        # Create groups from cluster labels
        groups = []
        for cluster_id in range(n_clusters):
            plane_indices = np.where(labels == cluster_id)[0]
            cluster_planes = [planes[i] for i in plane_indices]
            
            group = RoofGroup(planes=cluster_planes)
            for plane in group.planes:
                plane.group_id = group.id
            
            groups.append(group)
        
        if self.log_enabled:
            logger.info(f"Grouping: {len(planes)} planes → {len(groups)} groups")
        
        return groups
    
    def estimate_group_orientation(self, group: RoofGroup) -> float:
        """
        Estimate the orientation of a building group.
        
        This is a placeholder that could be enhanced with:
        - PCA on plane polygon shapes
        - Ridge detection
        - Eave alignment
        
        Args:
            group: RoofGroup to analyze
        
        Returns:
            Estimated orientation in degrees (0-360)
        """
        if not group.planes:
            return 0.0
        
        # Placeholder: use average orientation of polygons if available
        # For now, return 0 (will be enhanced in future phases)
        orientations = []
        for plane in group.planes:
            if plane.polygon and plane.polygon.vertices:
                # Compute orientation from first edge
                v0 = plane.polygon.vertices[0]
                v1 = plane.polygon.vertices[1] if len(plane.polygon.vertices) > 1 else v0
                dx = v1[0] - v0[0]
                dy = v1[1] - v0[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                orientations.append(angle)
        
        if orientations:
            # Average orientations (handle circular mean properly)
            avg_angle = np.mean(orientations)
            return float(avg_angle % 360)
        
        return 0.0
    
    def merge_groups(self, groups: List[RoofGroup], merge_distance: float) -> List[RoofGroup]:
        """
        Optionally merge nearby groups.
        
        This can be useful if DBSCAN produces oversegmentation.
        
        Args:
            groups: List of RoofGroup objects
            merge_distance: Distance threshold for merging groups
        
        Returns:
            Merged groups
        """
        if len(groups) <= 1:
            return groups
        
        # Compute group centers
        centers = np.array([g.building_center for g in groups])
        
        # Compute pairwise distances
        distances = cdist(centers, centers)
        
        # Find groups to merge
        merged = [False] * len(groups)
        result_groups = []
        
        for i in range(len(groups)):
            if merged[i]:
                continue
            
            # Find all groups within merge_distance
            close_indices = np.where(distances[i] < merge_distance)[0]
            
            if len(close_indices) > 1:
                # Merge groups
                merged_planes = []
                for j in close_indices:
                    merged_planes.extend(groups[j].planes)
                    merged[j] = True
                
                new_group = RoofGroup(planes=merged_planes)
                for plane in new_group.planes:
                    plane.group_id = new_group.id
                
                result_groups.append(new_group)
            else:
                # Keep single group as is
                result_groups.append(groups[i])
                merged[i] = True
        
        if self.log_enabled:
            logger.info(f"Merging: {len(groups)} → {len(result_groups)} groups")
        
        return result_groups
