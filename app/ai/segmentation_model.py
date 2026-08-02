"""
This module defines the abstract interface for all segmentation AI models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import uuid
import numpy as np

from app.ai.ai_model import AIModel
from app.ai.ai_result import SegmentationResult

class AbstractSegmentationModel(AIModel, ABC):
    """
    Abstract Base Class for vision-based AI models specifically for segmentation tasks.
    Extends AIModel with methods tailored for segmentation.
    """

    def __init__(self, model_name: str, version: str, model_id: Optional[uuid.UUID] = None):
        super().__init__(model_name, version, model_id)

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> List[SegmentationResult]:
        """
        Performs segmentation on the input image.

        Args:
            image (np.ndarray): The input image data (NumPy array, BGR format).
            **kwargs: Additional keyword arguments specific to the segmentation process.

        Returns:
            List[SegmentationResult]: A list of structured SegmentationResult objects.
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

    def predict(self, image: Any, **kwargs) -> Any:
        """
        The generic predict method from AIModel, which calls the specific segment method.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image for AbstractSegmentationModel must be a NumPy array.")
        return self.segment(image, **kwargs)

    def validate(self, data: Any) -> bool:
        """
        Validates the input data for the segmentation model.
        For vision models, this might check image format, dimensions, etc.
        """
        return isinstance(data, np.ndarray) and data.ndim >= 2
