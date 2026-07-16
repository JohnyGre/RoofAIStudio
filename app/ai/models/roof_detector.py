"""
This module implements a basic OpenCV-based roof detection.
"""

import uuid
from typing import Any, Dict, List, Union, Optional
import logging

import cv2
import numpy as np

from app.ai.models.vision_detector import VisionDetector
from app.ai.ai_result import DetectionResult, BoundingBox, SegmentationResult, GeometryPredictionResult
from app.core.logger import setup_logging
from app.ai.pipeline import (
    RoofPlaneDetector,
    RoofLineDetector,
    RoofFeatureDetector,
    RoofMaterialDetector,
    RoofDamageDetector
)

logger = setup_logging()

class RoofDetector(VisionDetector):
    """
    Facade roof detector that acts as an orchestrator delegating specific detection tasks
    to modular sub-detectors.
    """
    MODEL_NAME = "OpenCV_Roof_Detector"
    VERSION = "0.3.0-modular"

    def __init__(self, model_id: Optional[uuid.UUID] = None):
        super().__init__(self.MODEL_NAME, self.VERSION, model_id)
        # Register modular sub-detectors
        self._detectors = [
            RoofPlaneDetector(),
            RoofLineDetector(),
            RoofFeatureDetector(),
            RoofMaterialDetector(),
            RoofDamageDetector()
        ]
        self._model_info: Dict[str, Any] = {
            "name": self.MODEL_NAME,
            "version": self.VERSION,
            "description": "Orchestrator that delegates detection tasks to modular sub-detectors.",
            "input_requirements": "BGR NumPy array image",
            "output_format": "List[DetectionResult]",
            "trained_classes": ["roof_plane", "roof_line", "opening", "material", "damage"]
        }

    def load(self, model_path: str = "", device: str = "cpu", **kwargs) -> None:
        """
        Loads the detector. No external model required for OpenCV implementation.
        """
        logger.info(f"OpenCV RoofDetector facade '{self.model_name}' loaded (no external model file).")
        self._is_loaded = True

    def detect(self, image: np.ndarray, **kwargs) -> List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
        if not self.is_loaded:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded. Call load() first.")
        if not self.validate(image):
            raise ValueError("Invalid image input for detection.")

        # Run all modular sub-detectors and collect all DetectionResults
        results = []
        for detector in self._detectors:
            try:
                sub_results = detector.detect(image, **kwargs)
                results.extend(sub_results)
            except Exception as e:
                logger.error(f"Sub-detector {detector.__class__.__name__} failed: {e}")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        return self._model_info

    def validate(self, data: Any) -> bool:
        return isinstance(data, np.ndarray) and data.ndim >= 2 and data.shape[0] > 0 and data.shape[1] > 0
