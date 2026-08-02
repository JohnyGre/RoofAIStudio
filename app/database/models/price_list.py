"""
SQLAlchemy ORM models for PriceList and PriceItem entities.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Text, ForeignKey, DateTime, Float, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, BaseModel

class PriceList(Base, BaseModel):
    """
    Represents a specific price list from a supplier, valid for a certain period.
    """
    __tablename__ = "price_lists"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    supplier_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("suppliers.id"), nullable=False, index=True
    )
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    valid_to: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    currency: Mapped[str] = mapped_column(String(10), nullable=False, default="USD")

    supplier: Mapped["Supplier"] = relationship(back_populates="price_lists")
    price_items: Mapped[List["PriceItem"]] = relationship(back_populates="price_list", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("name", "supplier_id", name="uq_price_list_name_supplier"),
    )

class PriceItem(Base, BaseModel):
    """
    Represents a specific material's price within a price list.
    """
    __tablename__ = "price_items"

    material_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("materials.id"), nullable=False, index=True
    )
    price_list_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("price_lists.id"), nullable=False, index=True
    )
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    minimum_quantity: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    material: Mapped["Material"] = relationship() # No back_populates needed if not listing price items from Material
    price_list: Mapped["PriceList"] = relationship(back_populates="price_items")

    __table_args__ = (
        UniqueConstraint("material_id", "price_list_id", name="uq_price_item_material_pricelist"),
    )
