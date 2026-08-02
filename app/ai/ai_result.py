"""
This module defines data structures for various AI prediction results.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import uuid
import numpy as np

@dataclass(frozen=True)
class BoundingBox:
    """Represents a 2D bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

from abc import ABC

class BaseGeometry(ABC):
    """Abstract base class for all prediction geometries."""
    pass

@dataclass(frozen=True)
class BBoxGeometry(BaseGeometry):
    """Represents a 2D bounding box geometry."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

@dataclass(frozen=True)
class PolygonGeometry(BaseGeometry):
    """Represents a polygon geometry."""
    vertices: List[Tuple[float, float]]  # List of (x, y) coordinates

    def __post_init__(self):
        # Ensure immutable tuples
        object.__setattr__(self, 'vertices', tuple(self.vertices))

@dataclass(frozen=True)
class MaskGeometry(BaseGeometry):
    """Represents a binary or probability mask geometry."""
    mask: np.ndarray
    image_size: Optional[Tuple[int, int]] = None  # (width, height)

    def __post_init__(self):
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("Mask must be a NumPy array.")
        if self.mask.ndim not in [2, 3]:
            raise ValueError("Mask must be a 2D or 3D NumPy array.")

@dataclass(frozen=True)
class LineGeometry(BaseGeometry):
    """Represents a 2D line geometry."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]

@dataclass(frozen=True)
class DetectionResult:
    """
    Represents the result of an object detection/segmentation task.
    """
    class_name: str
    geometry: BaseGeometry
    confidence: float
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0.")
        
    @property
    def bounding_box(self) -> BoundingBox:
        """Backward compatibility helper to retrieve a BoundingBox representation."""
        if isinstance(self.geometry, BBoxGeometry):
            return BoundingBox(self.geometry.x_min, self.geometry.y_min, self.geometry.x_max, self.geometry.y_max)
        elif isinstance(self.geometry, PolygonGeometry):
            xs = [p[0] for p in self.geometry.vertices]
            ys = [p[1] for p in self.geometry.vertices]
            return BoundingBox(min(xs), min(ys), max(xs), max(ys))
        elif isinstance(self.geometry, LineGeometry):
            xs = [self.geometry.start_point[0], self.geometry.end_point[0]]
            ys = [self.geometry.start_point[1], self.geometry.end_point[1]]
            return BoundingBox(min(xs), min(ys), max(xs), max(ys))
        elif isinstance(self.geometry, MaskGeometry):
            # Calculate from mask coords where mask > 0.5
            pos = np.where(self.geometry.mask > 0.5)
            if pos[0].size > 0:
                y_min, x_min = float(np.min(pos[0])), float(np.min(pos[1]))
                y_max, x_max = float(np.max(pos[0])), float(np.max(pos[1]))
                return BoundingBox(x_min, y_min, x_max, y_max)
        return BoundingBox(0.0, 0.0, 0.0, 0.0)

@dataclass(frozen=True)
class SegmentationResult:
    """
    Represents the result of an image segmentation task.
    """
    mask: np.ndarray # Binary mask (e.g., 0s and 1s) or probability map
    class_name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    confidence: Optional[float] = None # Overall confidence for the mask
    image_size: Optional[tuple[int, int]] = None # (width, height) of the original image
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0.")
        if not isinstance(self.mask, np.ndarray):
            raise TypeError("Mask must be a NumPy array.")
        if self.mask.ndim not in [2, 3]: # 2D for grayscale, 3D for RGB/RGBA
            raise ValueError("Mask must be a 2D or 3D NumPy array.")

@dataclass(frozen=True)
class GeometryPredictionResult:
    """
    Represents a structured prediction of roof geometry.
    This could be a serialized representation of app.geometry.RoofGeometry
    or a simplified version.
    """
    predicted_geometry_data: Dict[str, Any] # Dictionary representing the predicted geometry
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0.")
        if not isinstance(self.predicted_geometry_data, dict):
            raise TypeError("predicted_geometry_data must be a dictionary.")