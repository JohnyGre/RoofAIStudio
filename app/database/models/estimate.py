"""
SQLAlchemy ORM models for Estimate and EstimateItem entities.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, BaseModel
from app.database.enums import EstimateStatus

class Estimate(Base, BaseModel):
    """Represents a cost estimate for a project."""
    __tablename__ = "estimates"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )
    project: Mapped["Project"] = relationship(back_populates="estimates")

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    total_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[EstimateStatus] = mapped_column(String(50), nullable=False, default=EstimateStatus.DRAFT)
    issue_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    valid_until_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    items: Mapped[List["EstimateItem"]] = relationship(back_populates="estimate", cascade="all, delete-orphan", lazy="joined")

    __table_args__ = (
        UniqueConstraint("name", "project_id", name="uq_estimate_name_project"),
    )

class EstimateItem(Base, BaseModel):
    """Represents a line item within an estimate."""
    __tablename__ = "estimate_items"

    estimate_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("estimates.id"), nullable=False, index=True
    )
    estimate: Mapped["Estimate"] = relationship(back_populates="items")

    material_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("materials.id"), nullable=False, index=True
    )
    material: Mapped["Material"] = relationship() # No back_populates needed if not listing estimate items from Material

    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    unit_price_at_time_of_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    total_price: Mapped[float] = mapped_column(Float, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
