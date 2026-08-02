"""
SQLAlchemy ORM models for Roof and RoofPhoto entities.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, BaseModel
from app.database.enums import PhotoType, RoofType

class Roof(Base, BaseModel):
    """Represents a single roof structure within a project."""
    __tablename__ = "roofs"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id"), nullable=False, index=True
    )
    project: Mapped["Project"] = relationship(back_populates="roofs")

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    roof_type: Mapped[RoofType] = mapped_column(String(50), nullable=False, default=RoofType.OTHER)
    area_sq_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_degrees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    photos: Mapped[List["RoofPhoto"]] = relationship(back_populates="roof", cascade="all, delete-orphan", lazy="joined")
    geometry: Mapped[Optional["RoofGeometry"]] = relationship(back_populates="roof", uselist=False, cascade="all, delete-orphan", lazy="joined")

    __table_args__ = (
        UniqueConstraint("name", "project_id", name="uq_roof_name_project"),
    )

class RoofPhoto(Base, BaseModel):
    """Stores information about photos associated with a roof."""
    __tablename__ = "roof_photos"

    roof_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roofs.id"), nullable=False, index=True
    )
    roof: Mapped["Roof"] = relationship(back_populates="photos")

    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    photo_type: Mapped[PhotoType] = mapped_column(String(50), nullable=False, default=PhotoType.OVERHEAD)
    capture_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    calibration_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON or similar
