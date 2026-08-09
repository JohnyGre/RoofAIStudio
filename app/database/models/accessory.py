"""
SQLAlchemy ORM model for RoofAccessory entity.
Roof accessories: ridge/hip strips, edge trims, ventilation, snow guards, etc.
"""
import uuid
from typing import Optional
from datetime import date

from sqlalchemy import Float, String, Text, Date, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BaseModel


class AccessoryCategory(Base, BaseModel):
    """Categories for roof accessories (hrebenace, ukoncovacie listy, etc.)."""
    __tablename__ = "accessory_categories"

    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    unit: Mapped[str] = mapped_column(String(20), nullable=False, default="ks")  # Default unit


class RoofAccessory(Base, BaseModel):
    """Represents a roof accessory product with pricing."""
    __tablename__ = "roof_accessories"

    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    manufacturer: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    manufacturer_web: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False, default="ks")
    min_slope_deg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    length_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight_kg_per_m2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    waste_percent: Mapped[float] = mapped_column(Float, nullable=False, default=5.0)
    supplier: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    supplier_web: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    supplier_region: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="EUR")
    valid_from: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self):
        return f"<RoofAccessory {self.manufacturer}/{self.name} {self.price:.2f} {self.currency}/{self.unit}>"
