"""
Roof material detector component.
Detects roof covering materials (e.g., asphalt shingle, tile, metal).
"""

from typing import List
import numpy as np
from app.ai.ai_result import DetectionResult
from app.ai.pipeline.base_detector import BaseSubDetector

class RoofMaterialDetector(BaseSubDetector):
    """Detects roof materials."""

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        return []
