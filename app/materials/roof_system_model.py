"""
This module defines data models for a complete roof system composition.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import uuid

from app.materials.material_model import Material

@dataclass(frozen=True)
class RoofLayer:
    """
    Represents a single layer within a roof system composition.
    """
    name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    material: Optional[Material] = None # The specific material used for this layer
    thickness: Optional[float] = None # e.g., in mm or inches
    description: Optional[str] = None
    order: int = 0 # Order in the layer stack (e.g., 0 for top layer)

@dataclass(frozen=True)
class RoofSystem:
    """
    Represents a complete roof composition, including multiple layers of materials.
    """
    name: str
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    description: Optional[str] = None
    layers: List[RoofLayer] = field(default_factory=list)

    def __post_init__(self):
        # Ensure layers are sorted by their order
        object.__setattr__(self, 'layers', sorted(self.layers, key=lambda layer: layer.order))
