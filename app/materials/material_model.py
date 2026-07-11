"""
This module defines the data models for materials, categories, units, and manufacturers.
These are dataclass representations, distinct from SQLAlchemy ORM models.
"""

from dataclasses import dataclass
from typing import Optional
import uuid

from app.database.enums import MaterialUnit as DBMaterialUnit # Import DB Enum for consistency

@dataclass(frozen=True)
class MaterialCategory:
    """Represents a category for materials."""
    id: uuid.UUID
    name: str
    description: Optional[str] = None

@dataclass(frozen=True)
class MaterialManufacturer:
    """Represents a manufacturer of materials."""
    id: uuid.UUID
    name: str
    contact_info: Optional[str] = None

@dataclass(frozen=True)
class Material:
    """
    Represents a specific material product with its properties.
    This is a domain model, not directly the SQLAlchemy ORM model.
    """
    id: uuid.UUID
    name: str
    category: MaterialCategory
    manufacturer: Optional[MaterialManufacturer]
    unit: DBMaterialUnit # Use the enum from the database layer
    price: float         # Price per unit
    coverage: Optional[float] = None # e.g., sq_ft per unit, or linear_ft per unit
    waste_factor: float = 0.0 # e.g., 0.05 for 5% waste
    description: Optional[str] = None
    sku: Optional[str] = None
    is_active: bool = True

    def __post_init__(self):
        if self.price < 0:
            raise ValueError("Material price cannot be negative.")
        if not (0 <= self.waste_factor < 1):
            raise ValueError("Waste factor must be between 0 and 1 (exclusive of 1).")


# Export the DB enum under the MaterialUnit name so external modules can import it from this module
MaterialUnit = DBMaterialUnit

__all__ = ["Material", "MaterialCategory", "MaterialManufacturer", "MaterialUnit"]
