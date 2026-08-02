"""
RoofPlane: Represents an individual roof plane (instance segmentation result).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import uuid
import numpy as np


@dataclass
class RoofPlanePolygon:
    """
    Refined polygon representation of a roof plane.
    
    Attributes:
        vertices: List of (x, y) tuples representing polygon vertices
        area: Polygon area in pixels
        perimeter: Polygon perimeter in pixels
        convex_hull: List of (x, y) tuples for the convex hull
        corner_count: Number of detected corners in the polygon
        simplified: Whether polygon has been simplified using Douglas-Peucker
    """
    vertices: List[Tuple[float, float]]
    area: float
    perimeter: float
    convex_hull: Optional[List[Tuple[float, float]]] = None
    corner_count: int = 0
    simplified: bool = False
    
    def to_numpy(self) -> np.ndarray:
        """Convert vertices to numpy array."""
        return np.array(self.vertices, dtype=np.float32)


@dataclass
class RoofPlane:
    """
    Represents a single roof plane detected by YOLO.
    
    This is the primary output of instance filtering and grouping stages.
    
    Attributes:
        id: Unique identifier for this plane
        mask: Binary segmentation mask (H x W)
        polygon: Refined polygon representation (computed lazily or during refinement)
        confidence: Detection confidence (0.0-1.0)
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        color_rgb: Average color of the plane (R, G, B)
        area_pixels: Mask area in pixels
        group_id: ID of the RoofGroup this plane belongs to (assigned during grouping)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mask: np.ndarray = field(default=None)
    polygon: Optional[RoofPlanePolygon] = None
    confidence: float = 0.5
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    color_rgb: Tuple[int, int, int] = (0, 0, 0)
    area_pixels: float = 0.0
    group_id: Optional[str] = None
    
    # Metadata for quality scoring
    solidity: float = 0.0  # Area / convex hull area (0.0-1.0)
    extent: float = 0.0    # Area / bounding box area (0.0-1.0)
    edge_consistency: float = 0.0  # Edge gradient consistency (0.0-1.0)
    convexity_ratio: float = 0.0   # Concavity measure (0.0 = convex, 1.0 = very concave)
    quality_score: float = 0.0     # Overall quality (0.0-1.0)
    
    def __post_init__(self):
        """Validate mask and compute area if mask is provided."""
        if self.mask is not None:
            if not isinstance(self.mask, np.ndarray):
                raise TypeError("Mask must be a NumPy array")
            if self.mask.ndim != 2:
                raise ValueError("Mask must be 2D (H x W)")
            self.area_pixels = np.sum(self.mask > 0)
    
    @property
    def has_polygon(self) -> bool:
        """Check if polygon has been computed."""
        return self.polygon is not None
    
    @property
    def centroid(self) -> Tuple[float, float]:
        """Compute centroid from bounding box."""
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
