"""
SQLAlchemy ORM model for the AITrainingSample entity.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database.base import Base, BaseModel

class AITrainingSample(Base, BaseModel):
    """
    Represents a single training sample used for AI model training,
    often derived from corrected AI predictions.
    """
    __tablename__ = "ai_training_samples"

    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    original_prediction: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON or serialized data of the original AI prediction
    corrected_geometry: Mapped[Optional[str]] = mapped_column(Text, nullable=True) # JSON or serialized data of the human-corrected geometry
    
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("roof_templates.id"), nullable=True, index=True
    )
    # template: Mapped["RoofTemplate"] = relationship() # If needed

    accuracy_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True) # Score indicating how close original prediction was to correction
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_used_for_training: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
