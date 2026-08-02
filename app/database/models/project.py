"""
SQLAlchemy ORM model for the Project entity.
"""

import uuid
from typing import List, Optional

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel
from app.database.enums import RoofStatus

class Project(Base, BaseModel):
    """Represents a single roof project."""
    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    address: Mapped[str] = mapped_column(String(500), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    state: Mapped[str] = mapped_column(String(100), nullable=False)
    zip_code: Mapped[str] = mapped_column(String(20), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False, default="USA")
    status: Mapped[RoofStatus] = mapped_column(String(50), nullable=False, default=RoofStatus.DRAFT)

    customer_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("customers.id"), nullable=False, index=True
    )
    customer: Mapped["Customer"] = relationship(back_populates="projects")
    roofs: Mapped[List["Roof"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    estimates: Mapped[List["Estimate"]] = relationship(back_populates="project", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("name", "customer_id", name="uq_project_name_customer"),
    )
