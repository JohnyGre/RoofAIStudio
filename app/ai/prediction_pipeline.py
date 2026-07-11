"""
This module defines the RoofAnalysisPipeline, orchestrating the full workflow
from image input to RoofGeometry output using AI.
"""

from typing import Any, Dict, List, Union, Optional, Tuple
import numpy as np
import uuid

from app.ai.ai_engine import AIEngine
from app.ai.ai_model import AIModel
from app.ai.ai_result import DetectionResult, GeometryPredictionResult
from app.ai.segmentation_result import SegmentationResult
from app.ai.geometry_converter import GeometryConverter
from app.ai.pipeline import CoreAIPipeline
from app.ai.roof_segmentation import RoofSegmentationService
from app.ai.segmentation_model import AbstractSegmentationModel
from app.core.logger import setup_logging
from app.core.image.image_processor import ImageProcessor
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel

logger = setup_logging()

class RoofAnalysisPipeline:
    """
    Orchestrates the complete workflow for analyzing a roof image using AI,
    from preprocessing to geometry conversion.
    """

    def __init__(self, ai_engine: AIEngine, geometry_converter: GeometryConverter):
        """
        Initializes the RoofAnalysisPipeline.

        Args:
            ai_engine (AIEngine): The AI Engine instance to use for model inference.
            geometry_converter (GeometryConverter): The service to convert AI results to geometry.
        """
        self._ai_engine = ai_engine
        self._geometry_converter = geometry_converter
        logger.info("RoofAnalysisPipeline initialized.")

    def analyze_image(
        self,
        image: np.ndarray,
        model_id: Optional[uuid.UUID] = None,
        model_name: Optional[str] = None,
        calibration: Optional[CalibrationModel] = None,
        **kwargs
    ) -> Tuple[RoofGeometry, List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]]:
        """
        Analyzes an input image to produce a RoofGeometry object and raw AI results.

        Workflow:
        1. Image preprocessing
        2. AI model prediction (detection or segmentation)
        3. Postprocessing
        4. Geometry conversion

        Args:
            image (np.ndarray): The input image (OpenCV BGR format).
            model_id (Optional[uuid.UUID]): The ID of the AI model to use.
            model_name (Optional[str]): The name of the AI model to use.
            calibration (Optional[CalibrationModel]): Calibration data for pixel-to-real-world conversion.
            **kwargs: Additional parameters for preprocessing, inference, postprocessing,
                      and geometry conversion stages.

        Returns:
            Tuple[RoofGeometry, List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]]:
                A tuple containing the predicted RoofGeometry and the raw AI results (for overlay).

        Raises:
            ValueError: If no model is specified or found, or if the model is not loaded.
            RuntimeError: If any stage of the pipeline fails.
        """
        logger.info(f"Starting roof analysis for image (shape: {image.shape})...")

        model = self._ai_engine.get_model(model_id=model_id, model_name=model_name)
        if not model:
            raise ValueError("No AI model specified or found for analysis.")
        if not model.is_loaded:
            raise ValueError(f"Model '{model.model_name}' is not loaded. Please load it first.")

        raw_ai_results: List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]
        roof_geometry: RoofGeometry

        # Determine if it's a segmentation model or a detection model
        if isinstance(model, AbstractSegmentationModel):
            # Use RoofSegmentationService for segmentation models
            segmentation_service = RoofSegmentationService(model)
            roof_geometry = segmentation_service.segment_and_convert_to_geometry(
                image=image,
                image_width=image.shape[1],
                image_height=image.shape[0],
                calibration=calibration,
                default_slope_degrees=kwargs.get("default_slope_degrees", 30.0),
                default_orientation_degrees=kwargs.get("default_orientation_degrees", 0.0),
                mask_processing_params=kwargs.get("mask_processing_params", {}),
                segmentation_params=kwargs.get("segmentation_params", {})
            )
            # To get raw_ai_results for overlay from a segmentation model,
            # we need to call its segment method directly.
            raw_ai_results = model.segment(image, **(kwargs.get("segmentation_params", {})))

        else:
            # Use CoreAIPipeline for other AI models (e.g., detection, direct geometry prediction)
            core_ai_pipeline = CoreAIPipeline(model)

            # Run the AI pipeline stages up to postprocessing
            raw_ai_results = core_ai_pipeline.run_pipeline(image, **kwargs)

            # Convert AI results to RoofGeometry using the GeometryConverter
            if all(isinstance(res, DetectionResult) for res in raw_ai_results):
                roof_geometry = self._geometry_converter.convert_detection_results_to_geometry(
                    detection_results=raw_ai_results,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                    calibration=calibration,
                    **kwargs.get("geometry_conversion_params", {})
                )
            elif all(isinstance(res, SegmentationResult) for res in raw_ai_results):
                roof_geometry = self._geometry_converter.convert_segmentation_results_to_geometry(
                    segmentation_results=raw_ai_results,
                    calibration=calibration,
                    **kwargs.get("geometry_conversion_params", {})
                )
            elif all(isinstance(res, GeometryPredictionResult) for res in raw_ai_results):
                if raw_ai_results:
                    roof_geometry = self._geometry_converter.convert_geometry_prediction_to_geometry(
                        geometry_prediction=raw_ai_results[0],
                        calibration=calibration
                    )
                else:
                    raise RuntimeError("No GeometryPredictionResult found in AI results.")
            else:
                logger.warning("Mixed or unsupported AI result types. Returning empty results.")
                raise TypeError("Unsupported or mixed AI result types for geometry conversion.")

        self._validate_output(roof_geometry)
        logger.info("Roof analysis completed successfully.")
        return roof_geometry, raw_ai_results

    def _validate_output(self, roof_geometry: RoofGeometry) -> None:
        """
        Validates the generated RoofGeometry object.

        Args:
            roof_geometry (RoofGeometry): The RoofGeometry object to validate.

        Raises:
            ValueError: If the RoofGeometry is invalid.
        """
        logger.debug("Validating RoofGeometry output...")
        try:
            roof_geometry.validate_geometry()
            logger.debug("RoofGeometry validation passed.")
        except ValueError as e:
            logger.error(f"Generated RoofGeometry failed validation: {e}")
            raise