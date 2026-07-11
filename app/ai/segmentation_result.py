"""
This module defines the data model for segmentation results.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import uuid
import numpy as np

@dataclass(frozen=True)
class SegmentationResult:
    """
    Represents the result of an image segmentation task.
    """
    mask: np.ndarray # Binary mask (e.g., 0s and 1s) or probability map
    class_name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4) # Moved to after required fields
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