"""
This module provides an ImageLoader service for loading and validating image files.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from PIL import Image

from app.core.image.image_model import ImageInfo
from app.core.image.image_metadata import ImageMetadataExtractor

class ImageLoader:
    """
    Service for loading image files, validating them, and extracting basic metadata.
    Supports PNG, JPG, JPEG, TIFF formats.
    """

    SUPPORTED_FORMATS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tiff", ".tif")

    @staticmethod
    def validate_image(file_path: Path) -> bool:
        """
        Validates if the given file path points to a supported and readable image.

        Args:
            file_path (Path): The path to the image file.

        Returns:
            bool: True if the image is valid and supported, False otherwise.
        """
        if not file_path.is_file():
            print(f"Validation Error: File not found at {file_path}")
            return False
        if file_path.suffix.lower() not in ImageLoader.SUPPORTED_FORMATS:
            print(f"Validation Error: Unsupported image format {file_path.suffix}. Supported: {ImageLoader.SUPPORTED_FORMATS}")
            return False
        try:
            # Try to open with Pillow to check integrity
            with Image.open(file_path) as img:
                img.verify() # Verify file integrity
            return True
        except Exception as e:
            print(f"Validation Error: Could not open or verify image {file_path}: {e}")
            return False

    @staticmethod
    def get_metadata(file_path: Path) -> Optional[ImageInfo]:
        """
        Extracts basic image metadata and returns an ImageInfo object.

        Args:
            file_path (Path): The path to the image file.

        Returns:
            Optional[ImageInfo]: An ImageInfo object if metadata can be extracted,
                                 None otherwise.
        """
        if not ImageLoader.validate_image(file_path):
            return None

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img_format = img.format

            created_date = ImageMetadataExtractor.get_creation_date(file_path)
            camera_info = ImageMetadataExtractor.get_camera_info(file_path)

            return ImageInfo(
                file_path=file_path,
                width=width,
                height=height,
                format=img_format if img_format else "UNKNOWN",
                created_date=created_date,
                camera_info=camera_info
            )
        except Exception as e:
            print(f"Error extracting metadata for {file_path}: {e}")
            return None

    @staticmethod
    def load_image(file_path: Path, as_opencv: bool = True) -> Optional[np.ndarray]:
        """
        Loads an image from the given file path.

        Args:
            file_path (Path): The path to the image file.
            as_opencv (bool): If True, loads the image as an OpenCV (NumPy) array (BGR format).
                              If False, loads as a Pillow Image object.

        Returns:
            Optional[np.ndarray] or Optional[Image.Image]: The loaded image,
                                                            or None if loading fails.
        """
        if not ImageLoader.validate_image(file_path):
            return None

        try:
            if as_opencv:
                # OpenCV loads images in BGR format by default
                img_cv = cv2.imread(str(file_path))
                if img_cv is None:
                    print(f"Error: OpenCV failed to load image {file_path}")
                    return None
                return img_cv
            else:
                # Pillow loads images in RGB format
                img_pil = Image.open(file_path)
                img_pil.load() # Ensure image data is loaded
                return img_pil
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None
