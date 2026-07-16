"""
Roof line detector component.
Detects roof edges, ridges, valleys, and eaves.
"""

from typing import List
import numpy as np
from app.ai.ai_result import DetectionResult, LineGeometry
from app.ai.pipeline.base_detector import BaseSubDetector

class RoofLineDetector(BaseSubDetector):
    """Detects roof lines (e.g., ridges, valleys, eaves)."""

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        # Return empty list or basic heuristics for now
        # Hough lines can be used as a helper/fallback here if needed
        return []
