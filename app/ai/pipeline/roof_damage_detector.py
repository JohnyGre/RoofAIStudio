"""
Roof damage detector component.
Detects roof damages (e.g., cracks, missing shingles, rust).
"""

from typing import List
import numpy as np
from app.ai.ai_result import DetectionResult
from app.ai.pipeline.base_detector import BaseSubDetector

class RoofDamageDetector(BaseSubDetector):
    """Detects roof damages."""

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        return []
