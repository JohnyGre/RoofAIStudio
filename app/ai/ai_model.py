"""
This module defines the abstract interface for AI models used in Roof AI Studio.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid

class AIModel(ABC):
    """
    Abstract Base Class for all AI models in Roof AI Studio.
    Defines the core interface that every AI model must implement.
    """

    def __init__(self, model_name: str, version: str, model_id: Optional[uuid.UUID] = None):
        self._model_id = model_id if model_id else uuid.uuid4()
        self._model_name = model_name
        self._version = version
        self._is_loaded = False

    @property
    def model_id(self) -> uuid.UUID:
        """Unique identifier for the model instance."""
        return self._model_id

    @property
    def model_name(self) -> str:
        """The human-readable name of the AI model."""
        return self._model_name

    @property
    def version(self) -> str:
        """The version of the AI model."""
        return self._version

    @property
    def is_loaded(self) -> bool:
        """Indicates if the model is currently loaded into memory."""
        return self._is_loaded

    @abstractmethod
    def load(self, model_path: str, device: str = "cpu", **kwargs) -> None:
        """
        Loads the AI model from the specified path into memory.

        Args:
            model_path (str): The file path to the model weights or configuration.
            device (str): The device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional keyword arguments specific to the model loading process.
        """
        pass

    @abstractmethod
    def predict(self, image: Any, **kwargs) -> Any:
        """
        Performs inference using the loaded AI model on the given input image.

        Args:
            image (Any): The input image data (e.g., NumPy array, PyTorch tensor).
            **kwargs: Additional keyword arguments specific to the prediction process.

        Returns:
            Any: The raw output from the AI model, to be processed by AIResult.
        """
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validates the input data or the model's output for correctness/consistency.

        Args:
            data (Any): The data to validate (e.g., input image, prediction output).

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        pass

    def unload(self) -> None:
        """
        Unloads the AI model from memory.
        Implementations should clear model weights and free resources.
        """
        self._is_loaded = False
        print(f"Model {self.model_name} (ID: {self.model_id}) unloaded.")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.model_name}', version='{self.version}', id='{self.model_id}')"
