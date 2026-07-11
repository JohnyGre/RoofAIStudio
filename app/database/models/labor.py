"""
SQLAlchemy ORM model for the LaborPrice entity.
"""

import uuid
from typing import Optional, Literal

from sqlalchemy import String, Text, Float, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BaseModel

class LaborPrice(Base, BaseModel):
    """
    Represents the pricing for different types of labor.
    """
    __tablename__ = "labor_prices"

    work_type: Mapped[str] = mapped_column(String(255), nullable=False, index=True) # e.g., "Roof Installation", "Demolition", "Repair"
    unit: Mapped[str] = mapped_column(String(50), nullable=False) # e.g., "m2", "linear_m", "hour", "piece"
    price: Mapped[float] = mapped_column(Float, nullable=False) # Price per unit
    difficulty_factor: Mapped[float] = mapped_column(Float, nullable=False, default=1.0) # Multiplier for complex jobs
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("work_type", "unit", name="uq_labor_worktype_unit"),
    )
