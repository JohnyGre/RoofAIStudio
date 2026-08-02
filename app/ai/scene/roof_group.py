"""
RoofGroup: Represents a group of roof planes belonging to the same building.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import uuid
import numpy as np

from .roof_plane import RoofPlane


@dataclass
class RoofGroup:
    """
    Represents a group of related roof planes (typically one building).
    
    Multiple roof planes are grouped based on spatial proximity and orientation.
    This is the primary output of the roof grouping stage.
    
    Attributes:
        id: Unique identifier for this group
        planes: List of RoofPlane objects in this group
        bounding_box: Combined bounding box (x_min, y_min, x_max, y_max)
        building_center: Estimated (x, y) center of the building
        plane_count: Number of planes in this group
        average_confidence: Average confidence of all planes
        estimated_orientation: Estimated roof orientation (degrees, 0-360)
        total_area_pixels: Sum of all plane areas
        convex_hull: Convex hull of all planes combined
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    planes: List[RoofPlane] = field(default_factory=list)
    bounding_box: Tuple[float, float, float, float] = (0, 0, 0, 0)
    building_center: Tuple[float, float] = (0, 0)
    plane_count: int = 0
    average_confidence: float = 0.0
    estimated_orientation: float = 0.0  # degrees, 0-360
    total_area_pixels: float = 0.0
    convex_hull: Optional[np.ndarray] = None
    
    # Quality metrics
    group_quality_score: float = 0.0  # 0.0-1.0, from averaging plane qualities
    
    def __post_init__(self):
        """Compute metrics from planes."""
        self.plane_count = len(self.planes)
        if self.plane_count > 0:
            self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute group-level metrics from planes."""
        if not self.planes:
            return
        
        # Average confidence
        self.average_confidence = np.mean([p.confidence for p in self.planes])
        
        # Sum area
        self.total_area_pixels = sum(p.area_pixels for p in self.planes)
        
        # Average quality
        self.group_quality_score = np.mean([p.quality_score for p in self.planes])
        
        # Bounding box (union of all plane bboxes)
        if self.planes:
            bboxes = [p.bbox for p in self.planes]
            x_mins = [b[0] for b in bboxes]
            y_mins = [b[1] for b in bboxes]
            x_maxs = [b[2] for b in bboxes]
            y_maxs = [b[3] for b in bboxes]
            self.bounding_box = (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))
        
        # Building center (centroid of all plane centroids)
        centroids = [p.centroid for p in self.planes]
        if centroids:
            cx = np.mean([c[0] for c in centroids])
            cy = np.mean([c[1] for c in centroids])
            self.building_center = (cx, cy)
    
    def add_plane(self, plane: RoofPlane):
        """Add a plane to this group and update metrics."""
        plane.group_id = self.id
        self.planes.append(plane)
        self._compute_metrics()
    
    def remove_plane(self, plane_id: str) -> bool:
        """Remove a plane by ID. Returns True if removed, False if not found."""
        initial_count = len(self.planes)
        self.planes = [p for p in self.planes if p.id != plane_id]
        if len(self.planes) < initial_count:
            self._compute_metrics()
            return True
        return False
    
    @property
    def width(self) -> float:
        """Bounding box width."""
        return self.bounding_box[2] - self.bounding_box[0]
    
    @property
    def height(self) -> float:
        """Bounding box height."""
        return self.bounding_box[3] - self.bounding_box[1]
    
    @property
    def min_confidence(self) -> float:
        """Minimum confidence among all planes."""
        return min(p.confidence for p in self.planes) if self.planes else 0.0
    
    @property
    def max_confidence(self) -> float:
        """Maximum confidence among all planes."""
        return max(p.confidence for p in self.planes) if self.planes else 0.0
