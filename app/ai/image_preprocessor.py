"""
Image Preprocessing Service

Handles image normalization, resizing, and format conversion.
Provides consistent preprocessing for detection and segmentation.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Centralizes image preprocessing operations.
    
    Handles:
    - Format conversion (BGR/RGB)
    - Resizing with aspect ratio preservation
    - Normalization
    - Data type conversion
    
    All parameters are configurable via config system.
    """

    def __init__(self, target_size: int = 640, device: str = "cpu"):
        """
        Initialize preprocessor.

        Args:
            target_size: Target image size for models (640, 1024, etc.)
            device: Device type ("cpu", "cuda", etc.) for format selection
        """
        self.target_size = target_size
        self.device = device

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate image format and shape.

        Args:
            image: Image array to validate

        Returns:
            True if image is valid BGR/RGB uint8 array.
        """
        if image is None or image.size == 0:
            logger.warning("Empty or None image")
            return False

        if image.ndim not in (2, 3):
            logger.warning(f"Invalid image shape: {image.shape}")
            return False

        if image.dtype != np.uint8:
            logger.warning(f"Expected uint8, got {image.dtype}")
            return False

        if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
            logger.warning(f"Invalid channel count: {image.shape[2]}")
            return False

        return True

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    @staticmethod
    def to_uint8(image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 if needed."""
        if image.dtype == np.uint8:
            return image

        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume 0-1 range for floats
            if image.max() <= 1.0:
                return (image * 255).astype(np.uint8)
            else:
                return image.astype(np.uint8)

        return image.astype(np.uint8)

    @staticmethod
    def to_float32(image: np.ndarray) -> np.ndarray:
        """Convert image to float32 normalized to [0, 1]."""
        if image.dtype == np.float32:
            return image
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)

    @staticmethod
    def normalize(image: np.ndarray, mean: Optional[Tuple] = None, std: Optional[Tuple] = None) -> np.ndarray:
        """
        Normalize image using mean and std.

        Args:
            image: Image as float32 in [0, 1]
            mean: Normalization mean per channel
            std: Normalization std per channel

        Returns:
            Normalized image
        """
        if mean is None or std is None:
            # Default ImageNet normalization
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        if image.ndim == 3:
            for i in range(image.shape[2]):
                image[..., i] = (image[..., i] - mean[i]) / std[i]

        return image

    def resize_preserve_aspect(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
        padding_value: int = 114
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Resize image preserving aspect ratio with padding.

        Args:
            image: Input image (H, W, C)
            target_size: Target size. If None, uses self.target_size
            padding_value: Value to use for padding

        Returns:
            Tuple of (resized_image, (scale_x, scale_y))
        """
        if target_size is None:
            target_size = self.target_size

        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)

        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        top = (target_size - new_h) // 2
        bottom = target_size - new_h - top
        left = (target_size - new_w) // 2
        right = target_size - new_w - left

        padded = cv2.copyMakeBorder(
            resized,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(padding_value, padding_value, padding_value)
        )

        logger.debug(f"Resized {image.shape} -> {resized.shape} (scale={scale:.3f}) -> padded {padded.shape}")

        return padded, (1.0 / scale, 1.0 / scale)

    def resize_to_size(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize to exact size (may distort aspect ratio).

        Args:
            image: Input image
            size: (height, width) target size

        Returns:
            Resized image
        """
        return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    def preprocess_for_yolo(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess image for YOLO inference.

        Args:
            image: Input BGR image uint8

        Returns:
            Tuple of (processed_image, scale_factors)
        """
        if not self.validate_image(image):
            raise ValueError(f"Invalid image format: {image.shape if image is not None else None}")

        # Resize with aspect preservation
        resized, scale = self.resize_preserve_aspect(image, self.target_size)

        logger.debug(f"YOLO preprocessing complete: {image.shape} -> {resized.shape}")

        return resized, scale

    def preprocess_for_sam(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for SAM inference.

        SAM works on original resolution, so only validate format.

        Args:
            image: Input BGR image uint8

        Returns:
            Converted to RGB image
        """
        if not self.validate_image(image):
            raise ValueError(f"Invalid image format: {image.shape if image is not None else None}")

        # SAM uses RGB
        rgb_image = self.bgr_to_rgb(image)

        logger.debug(f"SAM preprocessing complete: {rgb_image.shape}")

        return rgb_image

    def denormalize(
        self,
        image: np.ndarray,
        mean: Optional[Tuple] = None,
        std: Optional[Tuple] = None
    ) -> np.ndarray:
        """
        Denormalize image (reverse normalization).

        Args:
            image: Normalized image as float32
            mean: Mean used in normalization
            std: Std used in normalization

        Returns:
            Denormalized image
        """
        if mean is None or std is None:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        if image.ndim == 3:
            for i in range(image.shape[2]):
                image[..., i] = image[..., i] * std[i] + mean[i]

        return np.clip(image * 255, 0, 255).astype(np.uint8)
