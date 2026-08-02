"""
This module defines the core AI processing pipeline stages (preprocessing, inference, postprocessing).
"""

from typing import Any, Callable, List, Dict, Union, Optional
import numpy as np
import cv2 # For image processing in pipeline stages
import logging # Import logging module

from app.ai.ai_model import AIModel
from app.ai.ai_result import DetectionResult, GeometryPredictionResult, BoundingBox
from app.ai.segmentation_result import SegmentationResult # Use new SegmentationResult
from app.core.logger import setup_logging
from app.core.image.image_processor import ImageProcessor

logger: logging.Logger = setup_logging() # Assign the returned logger instance

class CoreAIPipeline:
    """
    Manages the sequential core stages of AI processing: preprocessing, inference, and postprocessing.
    It does NOT handle geometry conversion; that is a separate step.
    """

    def __init__(self, model: AIModel):
        """
        Initializes the CoreAIPipeline with a specific AI model.

        Args:
            model (AIModel): The AI model to be used for inference in this pipeline.
        """
        self.model = model
        if not self.model.is_loaded:
            logger.warning(f"AI model '{self.model.model_name}' is not loaded. "
                           "Ensure it's loaded before running the pipeline.")

    def _preprocess(self, image: np.ndarray, **kwargs) -> Any:
        """
        Stage 1: Preprocessing of the input image.
        This can include resizing, normalization, color space conversion, etc.

        Args:
            image (np.ndarray): The raw input image (OpenCV BGR format).
            **kwargs: Additional preprocessing parameters.

        Returns:
            Any: The preprocessed image data, ready for AI inference.
        """
        logger.debug("CoreAIPipeline: Preprocessing image...")
        # Example: Resize to a common input size for the model
        target_size = kwargs.get("target_size", (640, 640)) # Default target size
        resized_image = ImageProcessor.resize(image, target_size)
        if resized_image is None:
            raise ValueError("Image preprocessing (resize) failed.")

        # By default, return uint8 BGR resized image so OpenCV-based models can operate on CV_8U data.
        # If normalization is explicitly requested (e.g., for DL models), pass preprocess_params={'normalize': True}.
        if kwargs.get("normalize", False):
            normalized_image = resized_image.astype(np.float32) / 255.0
            return normalized_image
        else:
            # Ensure dtype is uint8 for OpenCV operations
            if resized_image.dtype != np.uint8:
                preprocessed_uint8 = (np.clip(resized_image, 0, 255)).astype(np.uint8)
            else:
                preprocessed_uint8 = resized_image
            return preprocessed_uint8

    def _inference(self, preprocessed_data: Any, **kwargs) -> Any:
        """
        Stage 2: AI inference using the loaded model.

        Args:
            preprocessed_data (Any): The output from the preprocessing stage.
            **kwargs: Additional inference parameters.

        Returns:
            Any: The raw output from the AI model's predict method.
        """
        logger.debug("CoreAIPipeline: Performing AI inference...")
        if not self.model.is_loaded:
            raise RuntimeError(f"AI model '{self.model.model_name}' is not loaded. Cannot perform inference.")
        return self.model.predict(preprocessed_data, **kwargs)

    def _postprocess(self, inference_output: Any, **kwargs) -> List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
        """
        Stage 3: Postprocessing of the AI model's raw output.
        This converts raw model outputs into structured AIResult dataclasses.

        Args:
            inference_output (Any): The raw output from the AI inference stage.
            **kwargs: Additional postprocessing parameters (e.g., confidence thresholds).

        Returns:
            List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]: A list of structured AI results.
        """
        logger.debug("CoreAIPipeline: Postprocessing AI inference output...")
        results: List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]] = []

        # This is a placeholder. Actual implementation depends heavily on the model's output format.
        # Example for a hypothetical object detection model output:
        if isinstance(inference_output, list) and all(isinstance(item, DetectionResult) for item in inference_output):
            # If the model's predict method already returns DetectionResults
            results.extend(inference_output)
        elif isinstance(inference_output, list) and all(isinstance(item, SegmentationResult) for item in inference_output):
            # If the model's predict method already returns SegmentationResults
            results.extend(inference_output)
        elif isinstance(inference_output, list) and all(isinstance(item, GeometryPredictionResult) for item in inference_output):
            # If the model's predict method already returns GeometryPredictionResults
            results.extend(inference_output)
        elif isinstance(inference_output, dict) and "boxes" in inference_output:
            for i in range(len(inference_output["boxes"])):
                bbox = inference_output["boxes"][i]
                score = inference_output["scores"][i]
                label = inference_output["labels"][i] # Assuming integer label
                class_name = kwargs.get("class_names", {}).get(label, f"Class_{label}")

                if score > kwargs.get("confidence_threshold", 0.5):
                    results.append(DetectionResult(
                        bounding_box=BoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], label=class_name),
                        confidence=float(score),
                        class_name=class_name
                    ))
        elif isinstance(inference_output, dict) and "masks" in inference_output:
            for i in range(len(inference_output["masks"])):
                mask = inference_output["masks"][i] # Assuming mask is a numpy array
                score = inference_output.get("scores", [None]*len(inference_output["masks"]))[i]
                label = inference_output.get("labels", [None]*len(inference_output["masks"]))[i]
                class_name = kwargs.get("class_names", {}).get(label, f"Class_{label}")

                results.append(SegmentationResult(
                    mask=mask,
                    class_name=class_name,
                    confidence=float(score) if score is not None else None,
                    image_size=kwargs.get("original_image_size") # Pass original image size for context
                ))
        else:
            logger.warning("Unknown AI model output format. Returning empty results.")

        return results

    def run_pipeline(self, image: np.ndarray, **kwargs) -> List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
        """
        Executes the core AI processing pipeline (preprocessing, inference, postprocessing).

        Args:
            image (np.ndarray): The raw input image (OpenCV BGR format).
            **kwargs: Parameters for different pipeline stages.

        Returns:
            List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
                A list of structured AI results from the postprocessing stage.
        """
        logger.info(f"Running CoreAIPipeline with model: {self.model.model_name}")

        # Store original image size for SegmentationResult
        original_image_size = (image.shape[1], image.shape[0]) # (width, height)
        kwargs["postprocess_params"] = kwargs.get("postprocess_params", {})
        kwargs["postprocess_params"]["original_image_size"] = original_image_size

        # Stage 1: Preprocessing
        preprocessed_data = self._preprocess(image, **kwargs.get("preprocess_params", {}))

        # Stage 2: AI Inference
        inference_output = self._inference(preprocessed_data, **kwargs.get("inference_params", {}))

        # Stage 3: Postprocessing
        ai_results = self._postprocess(inference_output, **kwargs.get("postprocess_params", {}))

        logger.info("CoreAIPipeline completed.")
        return ai_results
