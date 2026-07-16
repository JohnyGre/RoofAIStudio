"""
Roof feature detector component.
Detects roof features like chimneys, skylights, vents, etc.
"""

from typing import List
import numpy as np
from app.ai.ai_result import DetectionResult
from app.ai.pipeline.base_detector import BaseSubDetector

class RoofFeatureDetector(BaseSubDetector):
    """Detects roof features (e.g., openings, skylights, chimneys)."""

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        return []
