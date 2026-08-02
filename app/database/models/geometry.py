"""
SQLAlchemy ORM models for RoofGeometry, RoofPlane, RoofEdge, and RoofVertex entities.
"""

import uuid
from typing import List, Optional

from sqlalchemy import Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel
from app.database.enums import EdgeType

class RoofGeometry(Base, BaseModel):
    """Stores the geometric data for a roof."""
    __tablename__ = "roof_geometries"

    roof_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roofs.id"), nullable=False, unique=True, index=True
    )
    roof: Mapped["Roof"] = relationship(back_populates="geometry")

    total_area_sq_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_perimeter_lin_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    serialized_geometry: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    planes: Mapped[List["RoofPlane"]] = relationship(back_populates="geometry", cascade="all, delete-orphan", lazy="joined")
    edges: Mapped[List["RoofEdge"]] = relationship(back_populates="geometry", cascade="all, delete-orphan", lazy="joined")
    vertices: Mapped[List["RoofVertex"]] = relationship(back_populates="geometry", cascade="all, delete-orphan", lazy="joined")

class RoofPlane(Base, BaseModel):
    """Represents a single plane (face) of a roof."""
    __tablename__ = "roof_planes"

    geometry_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_geometries.id"), nullable=False, index=True
    )
    geometry: Mapped["RoofGeometry"] = relationship(back_populates="planes")

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    area_sq_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_degrees: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    vertex_ids: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON array of UUIDs

class RoofEdge(Base, BaseModel):
    """Represents an edge of a roof plane."""
    __tablename__ = "roof_edges"

    geometry_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_geometries.id"), nullable=False, index=True
    )
    geometry: Mapped["RoofGeometry"] = relationship(back_populates="edges")

    start_vertex_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("roof_vertices.id"), nullable=False)
    end_vertex_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("roof_vertices.id"), nullable=False)
    length_lin_ft: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    edge_type: Mapped[EdgeType] = mapped_column(String(50), nullable=False, default=EdgeType.UNKNOWN)

    start_vertex: Mapped["RoofVertex"] = relationship(
        foreign_keys=[start_vertex_id], back_populates="outgoing_edges"
    )
    end_vertex: Mapped["RoofVertex"] = relationship(
        foreign_keys=[end_vertex_id], back_populates="incoming_edges"
    )

class RoofVertex(Base, BaseModel):
    """Represents a vertex (point) in the 3D geometry of a roof."""
    __tablename__ = "roof_vertices"

    geometry_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_geometries.id"), nullable=False, index=True
    )
    geometry: Mapped["RoofGeometry"] = relationship(back_populates="vertices")

    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    z: Mapped[float] = mapped_column(Float, nullable=False)

    outgoing_edges: Mapped[List["RoofEdge"]] = relationship(
        foreign_keys="RoofEdge.start_vertex_id", back_populates="start_vertex"
    )
    incoming_edges: Mapped[List["RoofEdge"]] = relationship(
        foreign_keys="RoofEdge.end_vertex_id", back_populates="end_vertex"
    )
