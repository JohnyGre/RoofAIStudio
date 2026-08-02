"""
SQLAlchemy ORM model for the Customer entity.
"""

import uuid
from typing import List, Optional

from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel

class Customer(Base, BaseModel):
    """Represents a customer associated with projects."""
    __tablename__ = "customers"

    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    projects: Mapped[List["Project"]] = relationship(back_populates="customer", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("email", name="uq_customer_email"),
    )
