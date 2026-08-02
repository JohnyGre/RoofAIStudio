"""
Base Segmentation Service

Abstract base class for segmentation models (SAM, SAM2, etc.)
Provides common interface and shared functionality.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result from segmentation inference."""
    masks: np.ndarray  # (num_masks, H, W) boolean array
    scores: np.ndarray  # (num_masks,) confidence scores
    logits: Optional[np.ndarray] = None  # (num_masks, H, W) logits if available
    inference_time_ms: float = 0.0  # Inference latency
    model_type: str = ""  # Which model was used


class BaseSegmenter(ABC):
    """
    Abstract base class for segmentation models.
    
    Defines common interface for different segmentation backends (SAM, SAM2, etc.)
    with shared resource management and error handling.
    
    Subclasses must implement:
    - _load_model()
    - segment_from_point()
    - segment_from_box()
    - segment_auto()
    """

    def __init__(self, model_type: str = "vit_b", device: str = "auto"):
        """
        Initialize segmenter.

        Args:
            model_type: Model variant (e.g., "vit_b", "vit_h", "vit_l")
            device: Device to run on ("auto", "cpu", "cuda", etc.)
        """
        self.model_type = model_type
        self.device = device
        self.model = None
        self.predictor = None
        self.image_embedding = None
        self.current_image: Optional[np.ndarray] = None
        self._load_start_time = 0.0

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load model weights and initialize predictor.
        
        Must be implemented by subclasses.
        Should set self.model and self.predictor.
        """
        pass

    def load(self) -> None:
        """
        Load model with timing and error handling.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        if self.model is not None:
            logger.debug(f"{self.__class__.__name__} already loaded")
            return

        self._load_start_time = time.time()
        try:
            logger.info(f"Loading {self.__class__.__name__} ({self.model_type}) on {self.device}...")
            self._load_model()
            elapsed = time.time() - self._load_start_time
            logger.info(f"Model loaded in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load {self.__class__.__name__}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        if self.model is None:
            return

        try:
            logger.info(f"Unloading {self.__class__.__name__}...")
            if hasattr(self.model, "to"):
                self.model.to("cpu")
            del self.model
            self.model = None
            self.predictor = None
            self.image_embedding = None
            logger.info("Model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading model: {e}")

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the image for segmentation.
        
        Preprocesses and encodes the image so subsequent segmentation
        calls don't need to re-encode it.

        Args:
            image: RGB or BGR image (H, W, 3) uint8
            
        Raises:
            ValueError: If image format is invalid.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB/BGR image with shape (H, W, 3), got {image.shape}")

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Convert BGR to RGB if needed
            if image is not None:
                image_rgb = image[..., ::-1] if image.dtype == np.uint8 else image

            self.current_image = image
            start = time.time()
            self.predictor.set_image(image_rgb)
            self.image_embedding = self.predictor.get_image_embedding()
            elapsed = (time.time() - start) * 1000
            logger.debug(f"Image encoded in {elapsed:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to set image: {e}")
            raise

    @abstractmethod
    def segment_from_point(
        self,
        point_coords: List[Tuple[float, float]],
        point_labels: Optional[List[int]] = None,
        **kwargs
    ) -> SegmentationResult:
        """
        Segment from point prompts.

        Args:
            point_coords: List of (x, y) coordinates
            point_labels: List of labels (1=foreground, 0=background)
            **kwargs: Additional model-specific parameters

        Returns:
            SegmentationResult with masks, scores, etc.
        """
        pass

    @abstractmethod
    def segment_from_box(
        self,
        box: Tuple[float, float, float, float],
        **kwargs
    ) -> SegmentationResult:
        """
        Segment from bounding box prompt.

        Args:
            box: (x1, y1, x2, y2) box coordinates
            **kwargs: Additional model-specific parameters

        Returns:
            SegmentationResult with masks, scores, etc.
        """
        pass

    @abstractmethod
    def segment_auto(self, **kwargs) -> SegmentationResult:
        """
        Automatic segmentation without prompts.

        Returns:
            SegmentationResult with masks, scores, etc.
        """
        pass

    def validate_mask(self, mask: np.ndarray, min_area: int = 80) -> bool:
        """
        Validate mask quality.

        Args:
            mask: Binary mask array
            min_area: Minimum pixel area to accept

        Returns:
            True if mask meets quality criteria.
        """
        if mask.size == 0:
            return False
        if mask.sum() < min_area:
            logger.debug(f"Mask too small: {mask.sum()} < {min_area} pixels")
            return False
        return True

    def _measure_inference_time(self, start_time: float) -> float:
        """Measure inference latency in milliseconds."""
        return (time.time() - start_time) * 1000

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()
