"""
This module defines the abstract base interface for all pipeline sub-detectors.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from app.ai.ai_result import DetectionResult

class BaseSubDetector(ABC):
    """
    Abstract base class for specific AI detection tasks (e.g., planes, lines, features).
    """

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """
        Performs detection on the input image.

        Args:
            image (np.ndarray): Input BGR image.
            **kwargs: Configurable parameters for detection.

        Returns:
            List[DetectionResult]: Detected elements with structured geometry.
        """
        pass
