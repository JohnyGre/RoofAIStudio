"""
This module defines the interface and an SQLAlchemy implementation for Material data access.
"""

import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Type

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.database.models.material import (
    Material as ORMMaterial,
    MaterialCategory as ORMMaterialCategory,
    MaterialManufacturer as ORMMaterialManufacturer
)
from app.materials.material_model import (
    Material, MaterialCategory, MaterialManufacturer
)
from app.database.enums import MaterialUnit as DBMaterialUnit

class MaterialRepository(ABC):
    """
    Abstract base class defining the interface for Material data access operations.
    """

    @abstractmethod
    def add_material(self, material: Material) -> Material:
        """Adds a new material to the repository."""
        pass

    @abstractmethod
    def get_material(self, material_id: uuid.UUID) -> Optional[Material]:
        """Retrieves a material by its ID."""
        pass

    @abstractmethod
    def update_material(self, material: Material) -> Material:
        """Updates an existing material in the repository."""
        pass

    @abstractmethod
    def delete_material(self, material_id: uuid.UUID) -> None:
        """Deletes a material by its ID."""
        pass

    @abstractmethod
    def search_materials(
        self,
        name: Optional[str] = None,
        category_id: Optional[uuid.UUID] = None,
        manufacturer_id: Optional[uuid.UUID] = None,
        is_active: Optional[bool] = True
    ) -> List[Material]:
        """Searches for materials based on various criteria."""
        pass

    @abstractmethod
    def get_material_category(self, category_id: uuid.UUID) -> Optional[MaterialCategory]:
        """Retrieves a material category by its ID."""
        pass

    @abstractmethod
    def get_material_manufacturer(self, manufacturer_id: uuid.UUID) -> Optional[MaterialManufacturer]:
        """Retrieves a material manufacturer by its ID."""
        pass

    @abstractmethod
    def get_all_material_categories(self) -> List[MaterialCategory]:
        """Retrieves all material categories."""
        pass

    @abstractmethod
    def get_all_material_manufacturers(self) -> List[MaterialManufacturer]:
        """Retrieves all material manufacturers."""
        pass


class SQLAlchemyMaterialRepository(MaterialRepository):
    """
    SQLAlchemy implementation of the MaterialRepository interface.
    """

    def __init__(self, session: Session):
        self.session = session

    def _orm_to_domain_category(self, orm_category: ORMMaterialCategory) -> MaterialCategory:
        """Converts an ORM MaterialCategory to a domain MaterialCategory."""
        return MaterialCategory(
            id=orm_category.id,
            name=orm_category.name,
            description=orm_category.description
        )

    def _orm_to_domain_manufacturer(self, orm_manufacturer: ORMMaterialManufacturer) -> MaterialManufacturer:
        """Converts an ORM MaterialManufacturer to a domain MaterialManufacturer."""
        return MaterialManufacturer(
            id=orm_manufacturer.id,
            name=orm_manufacturer.name,
            contact_info=orm_manufacturer.contact_info
        )

    def _orm_to_domain_material(self, orm_material: ORMMaterial) -> Material:
        """Converts an ORM Material to a domain Material."""
        category = self._orm_to_domain_category(orm_material.category)
        manufacturer = self._orm_to_domain_manufacturer(orm_material.manufacturer) if orm_material.manufacturer else None
        return Material(
            id=orm_material.id,
            name=orm_material.name,
            category=category,
            manufacturer=manufacturer,
            unit=DBMaterialUnit(orm_material.unit_of_measure),
            price=orm_material.unit_cost,
            coverage=None, # ORM model does not currently have coverage
            waste_factor=0.0, # ORM model does not currently have waste_factor
            description=orm_material.description,
            sku=orm_material.sku,
            is_active=orm_material.is_active
        )

    def _domain_to_orm_material(self, material: Material, orm_material: Optional[ORMMaterial] = None) -> ORMMaterial:
        """Converts a domain Material to an ORM Material."""
        if orm_material is None:
            orm_material = ORMMaterial()
            orm_material.id = material.id

        orm_material.name = material.name
        orm_material.unit_of_measure = material.unit.value
        orm_material.unit_cost = material.price
        orm_material.description = material.description
        orm_material.sku = material.sku
        orm_material.is_active = material.is_active

        # Handle category
        orm_category = self.session.get(ORMMaterialCategory, material.category.id)
        if not orm_category:
            raise ValueError(f"MaterialCategory with ID {material.category.id} not found.")
        orm_material.category = orm_category

        # Handle manufacturer
        if material.manufacturer:
            orm_manufacturer = self.session.get(ORMMaterialManufacturer, material.manufacturer.id)
            if not orm_manufacturer:
                raise ValueError(f"MaterialManufacturer with ID {material.manufacturer.id} not found.")
            orm_material.manufacturer = orm_manufacturer
        else:
            orm_material.manufacturer = None

        return orm_material

    def add_material(self, material: Material) -> Material:
        orm_material = self._domain_to_orm_material(material)
        self.session.add(orm_material)
        self.session.flush() # To get ID if not already set
        return self._orm_to_domain_material(orm_material)

    def get_material(self, material_id: uuid.UUID) -> Optional[Material]:
        orm_material = self.session.get(ORMMaterial, material_id)
        if orm_material:
            return self._orm_to_domain_material(orm_material)
        return None

    def update_material(self, material: Material) -> Material:
        orm_material = self.session.get(ORMMaterial, material.id)
        if not orm_material:
            raise ValueError(f"Material with ID {material.id} not found for update.")
        self._domain_to_orm_material(material, orm_material)
        self.session.add(orm_material)
        self.session.flush()
        return self._orm_to_domain_material(orm_material)

    def delete_material(self, material_id: uuid.UUID) -> None:
        orm_material = self.session.get(ORMMaterial, material_id)
        if orm_material:
            self.session.delete(orm_material)
            self.session.flush()

    def search_materials(
        self,
        name: Optional[str] = None,
        category_id: Optional[uuid.UUID] = None,
        manufacturer_id: Optional[uuid.UUID] = None,
        is_active: Optional[bool] = True
    ) -> List[Material]:
        stmt = select(ORMMaterial)
        if name:
            stmt = stmt.where(ORMMaterial.name.ilike(f"%{name}%"))
        if category_id:
            stmt = stmt.where(ORMMaterial.category_id == category_id)
        if manufacturer_id:
            stmt = stmt.where(ORMMaterial.manufacturer_id == manufacturer_id)
        if is_active is not None:
            stmt = stmt.where(ORMMaterial.is_active == is_active)

        orm_materials = self.session.execute(stmt).scalars().all()
        return [self._orm_to_domain_material(m) for m in orm_materials]

    def get_material_category(self, category_id: uuid.UUID) -> Optional[MaterialCategory]:
        orm_category = self.session.get(ORMMaterialCategory, category_id)
        if orm_category:
            return self._orm_to_domain_category(orm_category)
        return None

    def get_material_manufacturer(self, manufacturer_id: uuid.UUID) -> Optional[MaterialManufacturer]:
        orm_manufacturer = self.session.get(ORMMaterialManufacturer, manufacturer_id)
        if orm_manufacturer:
            return self._orm_to_domain_manufacturer(orm_manufacturer)
        return None

    def get_all_material_categories(self) -> List[MaterialCategory]:
        stmt = select(ORMMaterialCategory)
        orm_categories = self.session.execute(stmt).scalars().all()
        return [self._orm_to_domain_category(c) for c in orm_categories]

    def get_all_material_manufacturers(self) -> List[MaterialManufacturer]:
        stmt = select(ORMMaterialManufacturer)
        orm_manufacturers = self.session.execute(stmt).scalars().all()
        return [self._orm_to_domain_manufacturer(m) for m in orm_manufacturers]
