"""
SQLAlchemy ORM model for ApplicationSettings entity.
"""

from typing import Optional

from sqlalchemy import String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, BaseModel

class ApplicationSettings(Base, BaseModel):
    """Stores application-wide settings and preferences."""
    __tablename__ = "application_settings"

    key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    setting_type: Mapped[str] = mapped_column(String(50), nullable=False, default="string") # e.g., "string", "int", "float", "bool", "json"
