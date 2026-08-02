"""
Detection Configuration Loader

Loads and manages AI detection parameters from config files.
Provides centralized configuration access throughout the detection pipeline.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """YOLO model configuration."""
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    nms_threshold: float = 0.7
    image_size: int = 640
    device: str = "auto"
    half_precision: bool = False
    max_batch_size: int = 8


@dataclass
class SAMConfig:
    """SAM segmentation configuration."""
    model_type: str = "vit_b"
    device: str = "auto"
    image_encoder_type: str = "default"


@dataclass
class MaskProcessingConfig:
    """Mask processing configuration."""
    binary_threshold: float = 0.5
    min_mask_area_pixels: int = 80
    morphology_kernel_size: int = 5
    morphology_iterations: int = 2
    contour_simplification_epsilon: float = 0.01


@dataclass
class EdgeDetectionConfig:
    """Edge detection configuration."""
    canny_threshold_low: int = 30
    canny_threshold_high: int = 90
    kernel_size: int = 3


@dataclass
class RoofPlaneDetectionConfig:
    """Roof plane detection configuration."""
    contour_quality_min_area: int = 500
    contour_quality_solidity_min: float = 0.7
    contour_quality_extent_min: float = 0.5
    contour_quality_hu_distance_max: float = 0.3
    watershed_markers: int = 100
    morphology_open_kernel_size: int = 5
    morphology_close_kernel_size: int = 7


@dataclass
class PostProcessingConfig:
    """Post-processing configuration."""
    nms_iou_threshold: float = 0.45
    merge_similar_planes: bool = True
    max_results: int = 15
    min_roof_quality_score: float = 0.25


@dataclass
class SAHIConfig:
    """SAHI large image processing configuration."""
    enabled: bool = False
    tile_size: int = 640
    tile_overlap_ratio: float = 0.2


@dataclass
class LineDetectionConfig:
    """Line detection configuration."""
    hough_rho_resolution: float = 1.0
    hough_theta_resolution: float = 3.14159 / 180  # 1 degree
    hough_threshold: int = 100
    hough_min_line_length: int = 100
    hough_max_line_gap: int = 20
    line_clustering_distance: int = 10


@dataclass
class PerformanceConfig:
    """Performance tuning configuration."""
    cache_models: bool = True
    preload_models_on_init: bool = False
    max_concurrent_inferences: int = 1


@dataclass
class DefaultGeometryConfig:
    """Default geometry parameters."""
    roof_slope_degrees: float = 30.0
    roof_orientation_degrees: float = 0.0
    building_height_meters: float = 8.0


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_model_loading: bool = True
    log_inference_time: bool = True
    log_preprocessing: bool = True
    log_exceptions: bool = True


class DetectionConfig:
    """
    Centralized configuration manager for AI detection subsystem.
    
    Loads configuration from detection.yaml and provides typed access
    to all detection parameters. Supports parameter override via kwargs.
    
    Example:
        config = DetectionConfig()
        yolo_conf = config.yolo.confidence_threshold
        config.update_yolo(confidence_threshold=0.5)
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to detection.yaml. If None, searches default locations.
        
        Raises:
            FileNotFoundError: If config file cannot be found.
            ValueError: If config file is invalid YAML.
        """
        self.config_path = self._find_config_file(config_path)
        self.raw_config: Dict[str, Any] = {}
        
        self.yolo = YOLOConfig()
        self.sam = SAMConfig()
        self.mask_processing = MaskProcessingConfig()
        self.edge_detection = EdgeDetectionConfig()
        self.roof_plane_detection = RoofPlaneDetectionConfig()
        self.post_processing = PostProcessingConfig()
        self.sahi = SAHIConfig()
        self.line_detection = LineDetectionConfig()
        self.performance = PerformanceConfig()
        self.default_geometry = DefaultGeometryConfig()
        self.logging = LoggingConfig()
        self.model_paths: Dict[str, str] = {}
        self.roof_colors: Dict[str, Dict[str, int]] = {}

        self._load_config()

    def _find_config_file(self, config_path: Optional[Path] = None) -> Path:
        """Find configuration file in default locations."""
        if config_path and config_path.exists():
            return config_path

        search_paths = [
            Path("config/detection.yaml"),
            Path(__file__).parent.parent.parent / "config" / "detection.yaml",
            Path.cwd() / "config" / "detection.yaml",
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found configuration at {path}")
                return path

        raise FileNotFoundError(
            f"detection.yaml not found in default locations: {search_paths}"
        )

    def _load_config(self) -> None:
        """Load and parse YAML configuration file."""
        if yaml is None:
            logger.warning("PyYAML not available. Using default configuration.")
            return

        try:
            with open(self.config_path, "r") as f:
                self.raw_config = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            raise ValueError(f"Invalid configuration file: {self.config_path}") from e

        self._parse_config()

    def _parse_config(self) -> None:
        """Parse raw YAML config into typed dataclass instances."""
        try:
            # Model paths
            models = self.raw_config.get("models", {})
            self.model_paths = {
                "yolo_building": models.get("yolo", {}).get("building_detector", "ai_models/yolov8n_building_seg.pt"),
                "yolo_coco": models.get("yolo", {}).get("coco_detector", "ai_models/yolov8n-seg.pt"),
                "yolo_roof": models.get("yolo", {}).get("roof_finetuned", "ai_models/roof_finetuned.pt"),
                "sam": models.get("sam", {}).get("vit_b", "ai_models/sam_vit_b_01ec64.pth"),
            }

            # YOLO config
            yolo_cfg = self.raw_config.get("yolo", {})
            self.yolo = YOLOConfig(
                confidence_threshold=yolo_cfg.get("confidence_threshold", 0.25),
                iou_threshold=yolo_cfg.get("iou_threshold", 0.45),
                nms_threshold=yolo_cfg.get("nms_threshold", 0.7),
                image_size=yolo_cfg.get("image_size", 640),
                device=yolo_cfg.get("device", "auto"),
                half_precision=yolo_cfg.get("half_precision", False),
                max_batch_size=yolo_cfg.get("max_batch_size", 8),
            )

            # SAM config
            sam_cfg = self.raw_config.get("sam", {})
            self.sam = SAMConfig(
                model_type=sam_cfg.get("model_type", "vit_b"),
                device=sam_cfg.get("device", "auto"),
                image_encoder_type=sam_cfg.get("image_encoder_type", "default"),
            )

            # Mask processing config
            mp_cfg = self.raw_config.get("mask_processing", {})
            self.mask_processing = MaskProcessingConfig(
                binary_threshold=mp_cfg.get("binary_threshold", 0.5),
                min_mask_area_pixels=mp_cfg.get("min_mask_area_pixels", 80),
                morphology_kernel_size=mp_cfg.get("morphology_kernel_size", 5),
                morphology_iterations=mp_cfg.get("morphology_iterations", 2),
                contour_simplification_epsilon=mp_cfg.get("contour_simplification_epsilon", 0.01),
            )

            # Edge detection config
            ed_cfg = self.raw_config.get("edge_detection", {})
            self.edge_detection = EdgeDetectionConfig(
                canny_threshold_low=ed_cfg.get("canny_threshold_low", 30),
                canny_threshold_high=ed_cfg.get("canny_threshold_high", 90),
                kernel_size=ed_cfg.get("kernel_size", 3),
            )

            # Roof plane detection config
            rpd_cfg = self.raw_config.get("roof_plane_detection", {})
            self.roof_plane_detection = RoofPlaneDetectionConfig(
                contour_quality_min_area=rpd_cfg.get("contour_quality_min_area", 500),
                contour_quality_solidity_min=rpd_cfg.get("contour_quality_solidity_min", 0.7),
                contour_quality_extent_min=rpd_cfg.get("contour_quality_extent_min", 0.5),
                contour_quality_hu_distance_max=rpd_cfg.get("contour_quality_hu_distance_max", 0.3),
                watershed_markers=rpd_cfg.get("watershed_markers", 100),
                morphology_open_kernel_size=rpd_cfg.get("morphology_open_kernel_size", 5),
                morphology_close_kernel_size=rpd_cfg.get("morphology_close_kernel_size", 7),
            )

            # Post-processing config
            pp_cfg = self.raw_config.get("post_processing", {})
            self.post_processing = PostProcessingConfig(
                nms_iou_threshold=pp_cfg.get("nms_iou_threshold", 0.45),
                merge_similar_planes=pp_cfg.get("merge_similar_planes", True),
                max_results=pp_cfg.get("max_results", 15),
                min_roof_quality_score=pp_cfg.get("min_roof_quality_score", 0.25),
            )

            # SAHI config
            sahi_cfg = self.raw_config.get("sahi", {})
            self.sahi = SAHIConfig(
                enabled=sahi_cfg.get("enabled", False),
                tile_size=sahi_cfg.get("tile_size", 640),
                tile_overlap_ratio=sahi_cfg.get("tile_overlap_ratio", 0.2),
            )

            # Line detection config
            ld_cfg = self.raw_config.get("line_detection", {})
            self.line_detection = LineDetectionConfig(
                hough_rho_resolution=ld_cfg.get("hough_rho_resolution", 1.0),
                hough_theta_resolution=ld_cfg.get("hough_theta_resolution", 3.14159 / 180),
                hough_threshold=ld_cfg.get("hough_threshold", 100),
                hough_min_line_length=ld_cfg.get("hough_min_line_length", 100),
                hough_max_line_gap=ld_cfg.get("hough_max_line_gap", 20),
                line_clustering_distance=ld_cfg.get("line_clustering_distance", 10),
            )

            # Performance config
            perf_cfg = self.raw_config.get("performance", {})
            self.performance = PerformanceConfig(
                cache_models=perf_cfg.get("cache_models", True),
                preload_models_on_init=perf_cfg.get("preload_models_on_init", False),
                max_concurrent_inferences=perf_cfg.get("max_concurrent_inferences", 1),
            )

            # Default geometry config
            dg_cfg = self.raw_config.get("default_geometry", {})
            self.default_geometry = DefaultGeometryConfig(
                roof_slope_degrees=dg_cfg.get("roof_slope_degrees", 30.0),
                roof_orientation_degrees=dg_cfg.get("roof_orientation_degrees", 0.0),
                building_height_meters=dg_cfg.get("building_height_meters", 8.0),
            )

            # Logging config
            log_cfg = self.raw_config.get("logging", {})
            self.logging = LoggingConfig(
                level=log_cfg.get("level", "INFO"),
                log_model_loading=log_cfg.get("log_model_loading", True),
                log_inference_time=log_cfg.get("log_inference_time", True),
                log_preprocessing=log_cfg.get("log_preprocessing", True),
                log_exceptions=log_cfg.get("log_exceptions", True),
            )

            # Roof colors
            self.roof_colors = self.raw_config.get("roof_colors", {})

            logger.info("Configuration parsed successfully")

        except Exception as e:
            logger.error(f"Failed to parse configuration: {e}")
            raise

    def get_model_path(self, model_name: str) -> str:
        """Get path to a specific model file."""
        return self.model_paths.get(model_name, "")

    def update_yolo(self, **kwargs) -> None:
        """Update YOLO configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.yolo, key):
                setattr(self.yolo, key, value)
                logger.info(f"Updated YOLO.{key} = {value}")

    def update_sam(self, **kwargs) -> None:
        """Update SAM configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.sam, key):
                setattr(self.sam, key, value)
                logger.info(f"Updated SAM.{key} = {value}")

    def update_post_processing(self, **kwargs) -> None:
        """Update post-processing configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.post_processing, key):
                setattr(self.post_processing, key, value)
                logger.info(f"Updated PostProcessing.{key} = {value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary representation."""
        return {
            "yolo": self.yolo.__dict__,
            "sam": self.sam.__dict__,
            "mask_processing": self.mask_processing.__dict__,
            "edge_detection": self.edge_detection.__dict__,
            "roof_plane_detection": self.roof_plane_detection.__dict__,
            "post_processing": self.post_processing.__dict__,
            "sahi": self.sahi.__dict__,
            "line_detection": self.line_detection.__dict__,
            "performance": self.performance.__dict__,
            "default_geometry": self.default_geometry.__dict__,
            "logging": self.logging.__dict__,
            "model_paths": self.model_paths,
        }


# Global configuration instance
_detection_config: Optional[DetectionConfig] = None


def get_detection_config() -> DetectionConfig:
    """Get or create the global detection configuration instance."""
    global _detection_config
    if _detection_config is None:
        _detection_config = DetectionConfig()
    return _detection_config


def load_detection_config(config_path: Optional[Path] = None) -> DetectionConfig:
    """Load detection configuration from file."""
    global _detection_config
    _detection_config = DetectionConfig(config_path)
    return _detection_config
