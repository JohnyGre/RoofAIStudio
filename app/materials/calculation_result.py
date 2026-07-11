"""
This module defines the data model for the result of a material calculation.
"""

from dataclasses import dataclass
from typing import Optional
import uuid

from app.database.enums import MaterialUnit

@dataclass(frozen=True)
class MaterialCalculationResult:
    """
    Represents the calculated quantities and cost for a single material.
    """
    material_id: uuid.UUID
    material_name: str
    quantity: float         # Base quantity needed (e.g., m2, linear m, pieces)
    unit: MaterialUnit      # Unit of the base quantity
    waste_quantity: float   # Additional quantity due to waste factor
    total_quantity: float   # quantity + waste_quantity
    estimated_cost: float   # Total estimated cost for total_quantity

    def __post_init__(self):
        if self.quantity < 0 or self.waste_quantity < 0 or self.total_quantity < 0 or self.estimated_cost < 0:
            raise ValueError("Quantities and cost cannot be negative.")
        if not isinstance(self.unit, MaterialUnit):
            raise TypeError("Unit must be an instance of MaterialUnit Enum.")
