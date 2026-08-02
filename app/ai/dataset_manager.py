"""
This module provides a DatasetManager for handling image datasets for AI testing and training.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import logging # Import logging module

from app.core.image.image_loader import ImageLoader
from app.core.logger import setup_logging
from app.core.app_paths import app_paths

logger: logging.Logger = setup_logging() # Assign the returned logger instance

class DatasetManager:
    """
    Manages image datasets, including adding, listing, and validating image files.
    It uses predefined directories for images, masks, and samples.
    """

    def __init__(self, base_data_path: Optional[Path] = None):
        """
        Initializes the DatasetManager with paths to image, mask, and sample directories.

        Args:
            base_data_path (Optional[Path]): The base path for data storage.
                                              Defaults to app_paths.data_dir.
        """
        self._base_data_path = base_data_path if base_data_path else app_paths.data_dir
        self._image_dir = self._base_data_path / "images"
        self._mask_dir = self._base_data_path / "masks"
        self._sample_dir = self._base_data_path / "samples"

        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._mask_dir.mkdir(parents=True, exist_ok=True)
        self._sample_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DatasetManager initialized. Image dir: {self._image_dir}")

    def add_image(self, source_path: Path, destination_name: Optional[str] = None) -> Optional[Path]:
        """
        Adds an image to the managed dataset by copying it to the images directory.

        Args:
            source_path (Path): The original path of the image file.
            destination_name (Optional[str]): The desired name for the copied file.
                                              If None, uses the original filename.

        Returns:
            Optional[Path]: The path to the copied image in the dataset, or None if validation fails.
        """
        if not ImageLoader.validate_image(source_path):
            logger.error(f"Failed to add image: {source_path} is not a valid image.")
            return None

        if destination_name:
            dest_path = self._image_dir / destination_name
        else:
            dest_path = self._image_dir / source_path.name
        
        try:
            dest_path.write_bytes(source_path.read_bytes())
            logger.info(f"Image added to dataset: {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Error copying image {source_path} to {dest_path}: {e}")
            return None

    def list_images(self) -> List[Path]:
        """
        Lists all valid image files in the managed images directory.

        Returns:
            List[Path]: A list of paths to valid image files.
        """
        valid_images: List[Path] = []
        for suffix in ImageLoader.SUPPORTED_FORMATS:
            valid_images.extend(self._image_dir.glob(f"*{suffix}"))
        
        # Filter out any invalid files that might have been copied
        return [img_path for img_path in valid_images if ImageLoader.validate_image(img_path)]

    def get_image_path(self, filename: str) -> Optional[Path]:
        """
        Returns the full path to an image in the dataset by its filename.

        Args:
            filename (str): The name of the image file.

        Returns:
            Optional[Path]: The full path, or None if not found.
        """
        image_path = self._image_dir / filename
        if image_path.is_file() and ImageLoader.validate_image(image_path):
            return image_path
        return None

    def validate_dataset(self) -> Tuple[int, int]:
        """
        Validates all images in the dataset and reports valid/invalid counts.

        Returns:
            Tuple[int, int]: A tuple (num_valid_images, num_invalid_images).
        """
        all_files = []
        for suffix in ImageLoader.SUPPORTED_FORMATS:
            all_files.extend(self._image_dir.glob(f"*{suffix}"))
        
        num_valid = 0
        num_invalid = 0
        for file_path in all_files:
            if ImageLoader.validate_image(file_path):
                num_valid += 1
            else:
                num_invalid += 1
                logger.warning(f"Invalid image found in dataset: {file_path}")
        
        logger.info(f"Dataset validation complete. Valid: {num_valid}, Invalid: {num_invalid}")
        return num_valid, num_invalid

    @property
    def image_dir(self) -> Path:
        return self._image_dir

    @property
    def mask_dir(self) -> Path:
        return self._mask_dir

    @property
    def sample_dir(self) -> Path:
        return self._sample_dir
