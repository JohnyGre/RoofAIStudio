"""
SQLAlchemy ORM models for Material, MaterialCategory, and MaterialManufacturer entities.
"""

import uuid
from typing import List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel
from app.database.enums import MaterialUnit

class MaterialCategory(Base, BaseModel):
    """Categorizes materials (e.g., Shingles, Flashing, Underlayment)."""
    __tablename__ = "material_categories"

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    materials: Mapped[List["Material"]] = relationship(back_populates="category", cascade="all, delete-orphan", lazy="joined")

class MaterialManufacturer(Base, BaseModel):
    """Represents a manufacturer of materials."""
    __tablename__ = "material_manufacturers"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    contact_info: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    materials: Mapped[List["Material"]] = relationship(back_populates="manufacturer", cascade="all, delete-orphan", lazy="joined")

class Material(Base, BaseModel):
    """Represents a specific material product."""
    __tablename__ = "materials"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    unit_cost: Mapped[float] = mapped_column(Float, nullable=False)
    unit_of_measure: Mapped[MaterialUnit] = mapped_column(String(50), nullable=False)
    sku: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    category_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("material_categories.id"), nullable=False, index=True
    )
    category: Mapped["MaterialCategory"] = relationship(back_populates="materials")

    manufacturer_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("material_manufacturers.id"), nullable=True, index=True
    )
    manufacturer: Mapped[Optional["MaterialManufacturer"]] = relationship(back_populates="materials")

    __table_args__ = (
        UniqueConstraint("name", "category_id", name="uq_material_name_category"),
    )
