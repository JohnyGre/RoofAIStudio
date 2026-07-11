"""
This module defines the main AI Engine service for Roof AI Studio.
It orchestrates AI model registration, prediction execution, and pipeline management.
"""

import uuid
from typing import Any, Dict, Optional, List

import numpy as np

from app.ai.ai_model import AIModel
from app.ai.model_registry import model_registry
from app.ai.pipeline import CoreAIPipeline # Renamed from AIPipeline
from app.ai.ai_result import DetectionResult, SegmentationResult, GeometryPredictionResult
from app.ai.geometry_converter import GeometryConverter # New import
from app.ai.prediction_pipeline import RoofAnalysisPipeline # New import
from app.core.logger import setup_logging
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel # New import

logger = setup_logging()

class AIEngine:
    """
    The central AI Engine responsible for managing and executing AI operations.
    It uses a ModelRegistry to keep track of available models and orchestrates
    prediction workflows through RoofAnalysisPipeline.
    """

    def __init__(self):
        """
        Initializes the AI Engine.
        """
        self._model_registry = model_registry # Use the singleton registry
        self._geometry_converter = GeometryConverter() # Instantiate GeometryConverter
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
        self._model_registry.unregister_model(model_id)

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

    def predict_geometry(
        self,
        image: np.ndarray,
        model_id: Optional[uuid.UUID] = None,
        model_name: Optional[str] = None,
        calibration: Optional[CalibrationModel] = None,
        **pipeline_kwargs
    ) -> RoofGeometry:
        """
        Executes the AI pipeline to predict roof geometry from an image.

        Args:
            image (np.ndarray): The input image (OpenCV BGR format).
            model_id (Optional[uuid.UUID]): The ID of the model to use for prediction.
            model_name (Optional[str]): The name of the model to use for prediction.
            calibration (Optional[CalibrationModel]): Calibration data for pixel-to-real-world conversion.
            **pipeline_kwargs: Additional keyword arguments to pass to the RoofAnalysisPipeline's analyze_image method.

        Returns:
            RoofGeometry: The predicted roof geometry.

        Raises:
            ValueError: If no model is specified or found, or if the model is not loaded.
        """
        # The RoofAnalysisPipeline now orchestrates the full process
        analysis_pipeline = RoofAnalysisPipeline(self, self._geometry_converter)
        return analysis_pipeline.analyze_image(
            image=image,
            model_id=model_id,
            model_name=model_name,
            calibration=calibration,
            **pipeline_kwargs
        )
