"""
SQLAlchemy ORM models for AIModel, AITrainingSample, and AIPrediction entities.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, BaseModel
from app.database.enums import AIModelType

class AIModel(Base, BaseModel):
    """Stores information about trained AI models."""
    __tablename__ = "ai_models"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    model_type: Mapped[AIModelType] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    training_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    performance_metrics: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON or similar

    predictions: Mapped[List["AIPrediction"]] = relationship(back_populates="ai_model", cascade="all, delete-orphan", lazy="joined")

class AITrainingSample(Base, BaseModel):
    """Represents a single training sample used for AI model training."""
    __tablename__ = "ai_training_samples"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    annotation_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True) # Path to annotation file (e.g., COCO, YOLO)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_used_for_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    __table_args__ = (
        UniqueConstraint("name", "image_path", name="uq_training_sample_name_path"),
    )

class AIPrediction(Base, BaseModel):
    """Stores results of AI model predictions on roof photos."""
    __tablename__ = "ai_predictions"

    ai_model_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("ai_models.id"), nullable=False, index=True
    )
    ai_model: Mapped["AIModel"] = relationship(back_populates="predictions")

    roof_photo_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("roof_photos.id"), nullable=False, index=True
    )
    roof_photo: Mapped["RoofPhoto"] = relationship() # No back_populates needed

    prediction_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    prediction_data: Mapped[str] = mapped_column(Text, nullable=False) # JSON or serialized prediction output
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
