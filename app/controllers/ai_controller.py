"""
This module defines the AIController, responsible for orchestrating AI analysis
and mediating between the UI and the AI pipeline.
"""

from typing import Optional, List, Union, Tuple
import uuid
import numpy as np

from PySide6.QtCore import QObject, Signal

from app.ai.ai_engine import AIEngine
from app.ai.prediction_pipeline import RoofAnalysisPipeline
from app.ai.geometry_converter import GeometryConverter
from app.ai.segmentation_result import SegmentationResult
from app.ai.ai_result import DetectionResult, GeometryPredictionResult
from app.ai.models.roof_detector import RoofDetector # For default model
from app.ai.models.yolo_adapter import YOLOSegmentationModel # For default model
from app.core.image.image_model import ImageInfo
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel
from app.core.logger import setup_logging
from app.core.app_paths import app_paths # For model path

logger = setup_logging()

class AIController(QObject):
    """
    Controller for managing AI-driven roof analysis.
    It orchestrates the AI pipeline, receives results, and prepares them for UI display.
    """

    analysis_started = Signal()
    analysis_completed = Signal(RoofGeometry, list) # Emits RoofGeometry and raw AI results (for overlay)
    error_occurred = Signal(str)
    status_message = Signal(str)

    def __init__(self, ai_engine: AIEngine, geometry_converter: GeometryConverter, parent: Optional[QObject] = None):
        """
        Initializes the AIController.

        Args:
            ai_engine (AIEngine): The AI Engine instance.
            geometry_converter (GeometryConverter): The GeometryConverter instance.
            parent (Optional[QObject]): The parent QObject.
        """
        super().__init__(parent)
        self._ai_engine = ai_engine
        self._geometry_converter = geometry_converter
        self._roof_analysis_pipeline = RoofAnalysisPipeline(ai_engine, geometry_converter)
        self._current_image_info: Optional[ImageInfo] = None
        self._current_image_data: Optional[np.ndarray] = None
        self._current_calibration: Optional[CalibrationModel] = None
        logger.info("AIController initialized.")

        # Register default models if not already registered (e.g., for testing)
        if not self._ai_engine.get_model(model_name=RoofDetector.MODEL_NAME):
            self._ai_engine.register_model(RoofDetector())
        # Example: Register YOLO model if available
        # if YOLOSegmentationModel is not None and not self._ai_engine.get_model(model_name=YOLOSegmentationModel.DEFAULT_MODEL_NAME):
        #     self._ai_engine.register_model(YOLOSegmentationModel())


    def set_current_image(self, image_data: np.ndarray, image_info: ImageInfo) -> None:
        """
        Sets the current image data and info for analysis.
        """
        self._current_image_data = image_data
        self._current_image_info = image_info
        logger.debug(f"AIController received image: {image_info.file_path.name}")

    def set_calibration_model(self, calibration: Optional[CalibrationModel]) -> None:
        """
        Sets the current calibration model for real-world conversions.

        Args:
            calibration (Optional[CalibrationModel]): The active calibration model, or None to clear.
        """
        self._current_calibration = calibration
        if calibration:
            logger.debug(f"AIController received calibration: {calibration.scale_factor_pixels_per_meter:.2f} px/m")
        else:
            logger.debug("AIController calibration cleared.")

    def analyze_roof(self, model_name: str = RoofDetector.MODEL_NAME) -> None: # Changed default model to RoofDetector.MODEL_NAME
        """
        Starts the AI-driven roof analysis process.

        Args:
            model_name (str): The name of the AI model to use for analysis.
        """
        if self._current_image_data is None or self._current_image_info is None:
            self.error_occurred.emit("No image loaded for AI analysis.")
            return
        if self._current_calibration is None:
            self.error_occurred.emit("No calibration set for AI analysis. Please calibrate the image first.")
            return

        self.analysis_started.emit()
        self.status_message.emit(f"Starting AI analysis with model: {model_name}...")
        logger.info(f"AI analysis started for {self._current_image_info.file_path.name} using model: {model_name}")

        try:
            # Ensure the model is loaded
            model = self._ai_engine.get_model(model_name=model_name)
            if not model:
                raise ValueError(f"AI model '{model_name}' not found in registry.")
            if not model.is_loaded:
                # For now, assume a default path for YOLO model weights
                # In a real app, this path would be configurable or stored in DB
                model_path = app_paths.ai_models_dir / "yolov8n-seg.pt"
                if not model_path.exists():
                     logger.warning(f"YOLO model weights not found at {model_path}. "
                                    "Please download a YOLOv8 segmentation model (e.g., yolov8n-seg.pt) "
                                    "and place it in this directory for full functionality.")
                     # Create a dummy file to prevent FileNotFoundError during load, but it won't work
                     model_path.touch() 
                self._ai_engine.load_model(model_name=model_name, model_path=str(model_path))

            # Run the full analysis pipeline
            roof_geometry, raw_ai_results = self._roof_analysis_pipeline.analyze_image(
                image=self._current_image_data,
                model_name=model_name,
                calibration=self._current_calibration,
                # Pass image dimensions for post-processing if needed by the model
                postprocess_params={"original_image_size": (self._current_image_info.width, self._current_image_info.height)}
            )

            self.analysis_completed.emit(roof_geometry, raw_ai_results)
            self.status_message.emit(f"AI analysis completed. Found {len(roof_geometry.planes)} roof planes.")
            logger.info(f"AI analysis completed for {self._current_image_info.file_path.name}.")

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            self.error_occurred.emit(f"AI analysis failed: {e}")

    # Removed display_result_on_canvas as its functionality is now handled by the signal connection in MainWindow.