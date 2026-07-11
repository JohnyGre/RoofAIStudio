"""
This module defines the main AI Engine service for Roof AI Studio.
It orchestrates AI model registration, prediction execution, and pipeline management.
"""

import uuid
from typing import Any, Dict, Optional, List, TYPE_CHECKING
import logging

import numpy as np

from app.ai.ai_model import AIModel
from app.ai.model_registry import model_registry
from app.core.logger import setup_logging

# Use TYPE_CHECKING to avoid circular imports for type hints
if TYPE_CHECKING:
    from app.ai.prediction_pipeline import RoofAnalysisPipeline
    from app.ai.geometry_converter import GeometryConverter
    from app.geometry.roof_geometry import RoofGeometry
    from app.geometry.calibration import CalibrationModel
    from app.ai.ai_result import DetectionResult, SegmentationResult, GeometryPredictionResult

logger: logging.Logger = setup_logging()

class AIEngine:
    """
    The central AI Engine responsible for managing and executing AI operations.
    It uses a ModelRegistry to keep track of available models.
    High-level prediction workflows are orchestrated by RoofAnalysisPipeline,
    which receives an AIEngine instance.
    """

    def __init__(self):
        """
        Initializes the AI Engine.
        """
        self._model_registry = model_registry
        logger.info("AI Engine initialized.")

    def register_model(self, model: AIModel) -> None:
        """
        Registers an AI model with the engine's model registry.

        Args:
            model (AIModel): The AI model instance to register.
        """
        self._model_registry.register_model(model)

    def unregister_model(self, model_id: uuid.UUID) -> None:
        """
        Unregisters an AI model from the engine's model registry.

        Args:
            model_id (uuid.UUID): The ID of the model to unregister.
        """
        if model_id in self._model_registry._models: # Access internal dict for direct removal
            model = self._model_registry._models.pop(model_id)
            if model.model_name in self._model_registry._name_to_id and self._model_registry._name_to_id[model.model_name] == model_id:
                del self._model_registry._name_to_id[model.model_name]
            logger.info(f"Unregistered AI model: {model.model_name} (ID: {model_id})")
        else:
            logger.warning(f"Attempted to unregister non-existent model with ID: {model_id}")

    def get_model(self, model_id: Optional[uuid.UUID] = None, model_name: Optional[str] = None) -> Optional[AIModel]:
        """
        Retrieves a registered AI model by its ID or name.

        Args:
            model_id (Optional[uuid.UUID]): The ID of the model.
            model_name (Optional[str]): The name of the model.

        Returns:
            Optional[AIModel]: The AI model instance, or None if not found.
        """
        if model_id:
            return self._model_registry.get_model_by_id(model_id)
        elif model_name:
            return self._model_registry.get_model_by_name(model_name)
        return None

    def list_available_models(self) -> List[AIModel]:
        """
        Lists all AI models currently registered with the engine.

        Returns:
            List[AIModel]: A list of registered AI model instances.
        """
        return self._model_registry.list_models()

    def load_model(self, model_id: Optional[uuid.UUID] = None, model_name: Optional[str] = None, model_path: str = "", device: str = "cpu", **kwargs) -> None:
        """
        Loads a specific AI model into memory.

        Args:
            model_id (Optional[uuid.UUID]): The ID of the model to load.
            model_name (Optional[str]): The name of the model to load.
            model_path (str): The file path to the model weights or configuration.
            device (str): The device to load the model on (e.g., "cpu", "cuda").
            **kwargs: Additional keyword arguments for model loading.

        Raises:
            ValueError: If the model is not found or loading fails.
        """
        model = self.get_model(model_id=model_id, model_name=model_name)
        if not model:
            raise ValueError(f"Model not found with ID '{model_id}' or name '{model_name}'.")
        
        try:
            model.load(model_path, device, **kwargs)
            logger.info(f"Model '{model.model_name}' (ID: {model.model_id}) loaded successfully on {device}.")
        except Exception as e:
            logger.error(f"Failed to load model '{model.model_name}' (ID: {model.model_id}): {e}")
            raise

    def unload_model(self, model_id: Optional[uuid.UUID] = None, model_name: Optional[str] = None) -> None:
        """
        Unloads a specific AI model from memory.

        Args:
            model_id (Optional[uuid.UUID]): The ID of the model to unload.
            model_name (Optional[str]): The name of the model to unload.
        """
        model = self.get_model(model_id=model_id, model_name=model_name)
        if model:
            model.unload()
        else:
            logger.warning(f"Attempted to unload non-existent model with ID '{model_id}' or name '{model_name}'.")

    # Removed predict_geometry method from AIEngine to break circular dependency.
    # This functionality is now handled by AIController via RoofAnalysisPipeline.
