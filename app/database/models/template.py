"""
SQLAlchemy ORM models for RoofTemplate and RoofTemplatePlane entities.
"""

import uuid
from typing import List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel
from app.database.enums import RoofType

class RoofTemplate(Base, BaseModel):
    """Defines reusable templates for common roof types or configurations."""
    __tablename__ = "roof_templates"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    roof_type: Mapped[RoofType] = mapped_column(String(50), nullable=False, default=RoofType.OTHER)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False) # For sharing templates

    template_planes: Mapped[List["RoofTemplatePlane"]] = relationship(back_populates="roof_template", cascade="all, delete-orphan", lazy="joined")

class RoofTemplatePlane(Base, BaseModel):
    """Represents a plane within a roof template."""
    __tablename__ = "roof_template_planes"

    roof_template_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_templates.id"), nullable=False, index=True
    )
    roof_template: Mapped["RoofTemplate"] = relationship(back_populates="template_planes")

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    relative_area_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # e.g., 0.5 for half the main roof area
    relative_pitch_degrees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    serialized_plane_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
