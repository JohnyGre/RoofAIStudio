"""
SQLAlchemy ORM models for the Roof Genome knowledge system.
These models describe roof geometry, construction features, and characteristics.
"""

import uuid
from typing import List, Optional

from sqlalchemy import String, Text, ForeignKey, Integer, Float, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel
from app.database.enums import RoofType as EnumRoofType # To avoid name collision with ORM model

class RoofType(Base, BaseModel):
    """
    Represents a general type of roof (e.g., Gable, Hip, Flat).
    This is the central entity for the Roof Genome.
    """
    __tablename__ = "roof_types"

    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, index=True) # Short code for the roof type
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships to associated genome components
    topology: Mapped[Optional["RoofTopology"]] = relationship(
        back_populates="roof_type", cascade="all, delete-orphan", uselist=False
    )
    geometry_rules: Mapped[Optional["RoofGeometryRules"]] = relationship(
        back_populates="roof_type", cascade="all, delete-orphan", uselist=False
    )
    features: Mapped[List["RoofFeature"]] = relationship(
        back_populates="roof_type", cascade="all, delete-orphan"
    )

class RoofTopology(Base, BaseModel):
    """
    Describes the topological characteristics of a specific roof type.
    """
    __tablename__ = "roof_topologies"

    roof_type_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_types.id"), nullable=False, unique=True, index=True
    )
    roof_type: Mapped["RoofType"] = relationship(back_populates="topology")

    plane_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ridge_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    valley_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hip_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gable_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

class RoofGeometryRules(Base, BaseModel):
    """
    Defines typical geometric rules and properties for a specific roof type.
    """
    __tablename__ = "roof_geometry_rules"

    roof_type_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_types.id"), nullable=False, unique=True, index=True
    )
    roof_type: Mapped["RoofType"] = relationship(back_populates="geometry_rules")

    min_pitch: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # Minimum pitch in degrees
    max_pitch: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # Maximum pitch in degrees
    symmetry: Mapped[Optional[str]] = mapped_column(String(50), nullable=True) # e.g., "bilateral", "radial", "asymmetric"
    complexity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # Numerical score for complexity

class RoofFeature(Base, BaseModel):
    """
    Lists common construction features associated with a specific roof type.
    """
    __tablename__ = "roof_features"

    roof_type_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_types.id"), nullable=False, index=True
    )
    roof_type: Mapped["RoofType"] = relationship(back_populates="features")

    feature_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True) # e.g., "chimney", "skylight", "dormer"
    is_common: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False) # Is this feature commonly found on this roof type?
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("roof_type_id", "feature_name", name="uq_roof_feature_type_name"),
    )
