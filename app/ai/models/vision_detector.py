"""
This module defines the abstract base class for all vision-based AI detectors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from app.ai.ai_model import AIModel
from app.ai.ai_result import DetectionResult, SegmentationResult, GeometryPredictionResult

class VisionDetector(AIModel, ABC):
    """
    Abstract Base Class for vision-based AI models (e.g., object detection, segmentation).
    Extends AIModel with methods specific to vision tasks.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
        """
        Performs detection/segmentation on the input image.

        Args:
            image (np.ndarray): The input image data (NumPy array, BGR format).
            **kwargs: Additional keyword arguments specific to the detection process.

        Returns:
            List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
                A list of structured AI results (e.g., bounding boxes, masks).
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns detailed information about the loaded model.

        Returns:
            Dict[str, Any]: A dictionary containing model details (e.g., architecture, trained classes).
        """
        pass

    # Implement AIModel's abstract methods
    def load(self, model_path: str, device: str = "cpu", **kwargs) -> None:
        """
        Loads the vision model. This method should be implemented by concrete detectors.
        """
        self._is_loaded = True
        print(f"VisionDetector '{self.model_name}' loaded (placeholder).")

    def predict(self, image: Any, **kwargs) -> Any:
        """
        The generic predict method from AIModel, which calls the specific detect method.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image for VisionDetector must be a NumPy array.")
        return self.detect(image, **kwargs)

    def validate(self, data: Any) -> bool:
        """
        Validates the input data for the detector.
        For vision models, this might check image format, dimensions, etc.
        """
        return isinstance(data, np.ndarray) and data.ndim >= 2
