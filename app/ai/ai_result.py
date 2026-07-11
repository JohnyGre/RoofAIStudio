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

@dataclass(frozen=True)
class DetectionResult:
    """
    Represents the result of an object detection task.
    """
    bounding_box: BoundingBox
    confidence: float
    class_name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0.")

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