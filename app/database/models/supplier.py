"""
SQLAlchemy ORM model for the Supplier entity.
"""

import uuid
from typing import List, Optional

from sqlalchemy import String, Text
from sqlalchemy.orm import relationship, Mapped, mapped_column

from app.database.base import Base, BaseModel

class Supplier(Base, BaseModel):
    """
    Represents a supplier of materials or services.
    """
    __tablename__ = "suppliers"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    contact: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # e.g., contact person, phone, email
    website: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    price_lists: Mapped[List["PriceList"]] = relationship(back_populates="supplier", cascade="all, delete-orphan")
