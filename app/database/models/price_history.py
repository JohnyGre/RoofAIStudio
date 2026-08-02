"""
SQLAlchemy ORM model for the PriceHistory entity.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, BaseModel

class PriceHistory(Base, BaseModel):
    """
    Tracks changes in material prices over time.
    """
    __tablename__ = "price_history"

    material_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("materials.id"), nullable=False, index=True
    )
    # material: Mapped["Material"] = relationship() # If needed for direct navigation

    old_price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    new_price: Mapped[float] = mapped_column(Float, nullable=False)
    changed_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True) # e.g., "Manual Update", "Supplier Price List"
