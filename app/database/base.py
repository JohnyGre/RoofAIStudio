"""
This module defines the base classes and common utilities for SQLAlchemy ORM models.
It includes a declarative base and a BaseModel mixin for common fields like
ID, creation timestamp, and update timestamp.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.sqlite import UUID as SQLiteUUID
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy.sql import func

# Define the declarative base for all ORM models
Base: Any = declarative_base()

class BaseModel:
    """
    Mixin for common model attributes: id, created_at, updated_at.
    """
    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        SQLiteUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        unique=True,
        nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    def __repr__(self) -> str:
        """
        Provides a string representation of the model instance.
        """
        return f"<{self.__class__.__name__}(id={self.id})>"
