"""
SQLAlchemy ORM models for RoofTemplate and RoofTemplatePlane entities.
"""

import uuid
from typing import List, Optional

from sqlalchemy import Boolean, Float, ForeignKey, String, Text, UniqueConstraint, Integer
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel
from app.database.enums import RoofType

class RoofTemplate(Base, BaseModel):
    """
    Defines reusable templates for common roof types or configurations.
    This model stores a generalized representation of a roof shape.
    """
    __tablename__ = "roof_templates"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    roof_type: Mapped[RoofType] = mapped_column(String(50), nullable=False, default=RoofType.OTHER, index=True)
    complexity_level: Mapped[Optional[str]] = mapped_column(String(100), nullable=True) # e.g., "Simple", "Moderate", "Complex"

    # Geometry signature for quick comparison
    number_of_planes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    number_of_ridges: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    number_of_valleys: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    number_of_hips: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    number_of_openings: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationship to RoofTemplatePlane (if detailed plane info is needed for templates)
    template_planes: Mapped[List["RoofTemplatePlane"]] = relationship(
        back_populates="roof_template", cascade="all, delete-orphan", lazy="joined"
    )

class RoofTemplatePlane(Base, BaseModel):
    """
    Represents a plane within a roof template.
    This model was already defined in the previous step, ensuring consistency.
    """
    __tablename__ = "roof_template_planes"

    roof_template_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_templates.id"), nullable=False, index=True
    )
    roof_template: Mapped["RoofTemplate"] = relationship(back_populates="template_planes")

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    relative_area_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relative_pitch_degrees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    serialized_plane_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
