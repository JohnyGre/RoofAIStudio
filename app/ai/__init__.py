from .ai_model import AIModel
from .ai_result import DetectionResult, GeometryPredictionResult, BoundingBox
from .segmentation_result import SegmentationResult
from .model_registry import ModelRegistry, model_registry
from .pipeline import CoreAIPipeline
from .prediction_pipeline import RoofAnalysisPipeline
from .geometry_converter import GeometryConverter
from .ai_engine import AIEngine
from .segmentation_model import AbstractSegmentationModel
from .mask_processor import MaskProcessor
from .roof_segmentation import RoofSegmentationService
from .dataset_manager import DatasetManager # New import
from .ai_test_runner import AITestRunner # New import

# Import concrete models to ensure they are registered
from .models.vision_detector import VisionDetector
from .models.roof_detector import RoofDetector
from .models.yolo_adapter import YOLOSegmentationModel

# Register default models when the AI package is imported
# This ensures they are available in the singleton model_registry
model_registry.register_model(RoofDetector())
# Register YOLO Segmentation Model (example, might be conditional or loaded dynamically)
# model_registry.register_model(YOLOSegmentationModel())
