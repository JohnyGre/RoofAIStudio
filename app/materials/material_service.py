"""
This module defines the MaterialService for business logic related to materials.
"""

import uuid
from typing import Optional, List

from app.materials.material_model import Material, MaterialCategory, MaterialManufacturer
from app.materials.material_repository import MaterialRepository
from app.database.enums import MaterialUnit as DBMaterialUnit

class MaterialService:
    """
    Service layer for handling material-related business logic, such as quantity calculations
    and cost estimations.
    """

    def __init__(self, repository: MaterialRepository):
        self.repository = repository

    def calculate_material_quantity(
        self,
        material_id: uuid.UUID,
        required_area_sq_m: Optional[float] = None,
        required_length_m: Optional[float] = None,
        required_pieces: Optional[int] = None
    ) -> float:
        """
        Calculates the required quantity of a material based on area, length, or pieces.

        Args:
            material_id (uuid.UUID): The ID of the material.
            required_area_sq_m (Optional[float]): Required area in square meters.
            required_length_m (Optional[float]): Required length in meters.
            required_pieces (Optional[int]): Required number of individual pieces.

        Returns:
            float: The calculated quantity in the material's native unit.

        Raises:
            ValueError: If insufficient information is provided or material coverage is missing.
        """
        material = self.repository.get_material(material_id)
        if not material:
            raise ValueError(f"Material with ID {material_id} not found.")

        quantity = 0.0
        if material.unit == DBMaterialUnit.SQUARE_FOOT:
            if required_area_sq_m is None:
                raise ValueError("Required area in square meters must be provided for square foot materials.")
            if material.coverage is None or material.coverage <= 0:
                raise ValueError(f"Material {material.name} (ID: {material_id}) has no defined coverage per unit.")
            # Convert required_area_sq_m to sq_ft
            required_area_sq_ft = required_area_sq_m * 10.7639 # 1 sq meter = 10.7639 sq feet
            quantity = required_area_sq_ft / material.coverage
        elif material.unit == DBMaterialUnit.LINEAR_FOOT:
            if required_length_m is None:
                raise ValueError("Required length in meters must be provided for linear foot materials.")
            if material.coverage is None or material.coverage <= 0:
                raise ValueError(f"Material {material.name} (ID: {material_id}) has no defined coverage per unit.")
            # Convert required_length_m to linear_ft
            required_length_ft = required_length_m * 3.28084 # 1 meter = 3.28084 feet
            quantity = required_length_ft / material.coverage
        elif material.unit == DBMaterialUnit.EACH:
            if required_pieces is None:
                raise ValueError("Required number of pieces must be provided for 'each' materials.")
            quantity = float(required_pieces)
        elif material.unit == DBMaterialUnit.BUNDLE:
            if required_area_sq_m is None:
                raise ValueError("Required area in square meters must be provided for bundle materials.")
            if material.coverage is None or material.coverage <= 0:
                raise ValueError(f"Material {material.name} (ID: {material_id}) has no defined coverage per bundle.")
            # Assuming coverage is in sq_ft per bundle
            required_area_sq_ft = required_area_sq_m * 10.7639
            quantity = required_area_sq_ft / material.coverage
        else:
            raise ValueError(f"Unsupported material unit for quantity calculation: {material.unit}")

        return quantity

    def apply_waste_factor(self, quantity: float, material_id: uuid.UUID) -> float:
        """
        Applies the material's waste factor to a given quantity.

        Args:
            quantity (float): The base quantity of the material.
            material_id (uuid.UUID): The ID of the material.

        Returns:
            float: The quantity including the waste factor.

        Raises:
            ValueError: If the material is not found.
        """
        material = self.repository.get_material(material_id)
        if not material:
            raise ValueError(f"Material with ID {material_id} not found.")
        return quantity * (1 + material.waste_factor)

    def calculate_cost(self, material_id: uuid.UUID, quantity: float) -> float:
        """
        Calculates the total cost for a given quantity of material.

        Args:
            material_id (uuid.UUID): The ID of the material.
            quantity (float): The quantity of the material.

        Returns:
            float: The total cost.

        Raises:
            ValueError: If the material is not found or quantity is negative.
        """
        if quantity < 0:
            raise ValueError("Quantity cannot be negative.")
        material = self.repository.get_material(material_id)
        if not material:
            raise ValueError(f"Material with ID {material_id} not found.")
        return material.price * quantity

    def get_material_by_id(self, material_id: uuid.UUID) -> Optional[Material]:
        """Retrieves a material by its ID."""
        return self.repository.get_material(material_id)

    def get_all_materials(self) -> List[Material]:
        """Retrieves all active materials."""
        return self.repository.search_materials(is_active=True)

    def get_materials_by_category(self, category_id: uuid.UUID) -> List[Material]:
        """Retrieves materials belonging to a specific category."""
        return self.repository.search_materials(category_id=category_id)

    def get_all_categories(self) -> List[MaterialCategory]:
        """Retrieves all material categories."""
        return self.repository.get_all_material_categories()

    def get_all_manufacturers(self) -> List[MaterialManufacturer]:
        """Retrieves all material manufacturers."""
        return self.repository.get_all_material_manufacturers()
