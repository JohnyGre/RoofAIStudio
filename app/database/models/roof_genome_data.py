"""
SQLAlchemy ORM model for the RoofGenome entity, representing a numerical fingerprint
of a roof's geometry for similarity search.
"""

import uuid
from typing import Optional

from sqlalchemy import Float, String, Text, ForeignKey, LargeBinary, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BaseModel

class RoofGenome(Base, BaseModel):
    """
    Represents a numerical fingerprint or signature of a roof's geometry.
    Designed for efficient comparison and AI similarity searches.
    """
    __tablename__ = "roof_genomes"

    # Optional foreign keys to link to a specific roof or a template
    roof_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("roofs.id"), nullable=True, index=True
    )
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("roof_templates.id"), nullable=True, index=True
    )

    # Geometry signature (numerical features)
    plane_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ridge_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    valley_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hip_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    opening_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    average_slope: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    symmetry_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    complexity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Feature vector for AI similarity search (e.g., embedding from a neural network)
    # Stored as binary data, can be deserialized into a NumPy array
    feature_vector: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    # Relationships (optional, for easier navigation)
    # roof: Mapped["Roof"] = relationship(back_populates="genome") # If Roof model had a genome field
    # template: Mapped["RoofTemplate"] = relationship(back_populates="genomes") # If RoofTemplate had a genomes field
