"""
This module defines the ImageController, responsible for handling image-related
operations and mediating between the UI and the image processing core.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Signal, QObject

from app.core.image.image_loader import ImageLoader
from app.core.image.image_model import ImageInfo
from app.core.logger import setup_logging

logger = setup_logging()

class ImageController(QObject):
    """
    Controller for managing image loading, validation, and display.
    It acts as an intermediary between the UI (e.g., MainWindow, RoofCanvas)
    and the core image processing modules.
    """

    # Signals to communicate with the UI
    image_loaded = Signal(QPixmap, ImageInfo)
    image_cleared = Signal()
    error_occurred = Signal(str)
    status_message = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        """
        Initializes the ImageController.

        Args:
            parent (Optional[QObject]): The parent QObject.
        """
        super().__init__(parent)
        self._current_image_info: Optional[ImageInfo] = None
        self._current_image_data: Optional[np.ndarray] = None # Store OpenCV format image

    @property
    def current_image_info(self) -> Optional[ImageInfo]:
        """Returns the ImageInfo of the currently loaded image."""
        return self._current_image_info

    @property
    def current_image_data(self) -> Optional[np.ndarray]:
        """Returns the raw NumPy array (OpenCV BGR) of the currently loaded image."""
        return self._current_image_data

    def load_image(self, file_path: Path) -> None:
        """
        Loads an image from the given file path, validates it, and prepares it for display.

        Args:
            file_path (Path): The path to the image file.
        """
        self.status_message.emit(f"Loading image: {file_path.name}...")
        logger.info(f"Attempting to load image: {file_path}")

        if not ImageLoader.validate_image(file_path):
            error_msg = f"Invalid or unsupported image file: {file_path.name}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.status_message.emit("Image loading failed.")
            return

        image_data_cv = ImageLoader.load_image(file_path, as_opencv=True)
        if image_data_cv is None:
            error_msg = f"Failed to load image data for: {file_path.name}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self.status_message.emit("Image loading failed.")
            return

        image_info = ImageLoader.get_metadata(file_path)
        if image_info is None:
            logger.warning(f"Could not retrieve full metadata for {file_path.name}, proceeding with basic info.")
            # Create a minimal ImageInfo if metadata extraction fails
            h, w, _ = image_data_cv.shape
            image_info = ImageInfo(file_path=file_path, width=w, height=h, format=file_path.suffix.lstrip('.').upper())

        self._current_image_data = image_data_cv
        self._current_image_info = image_info
        
        self.display_image()
        self.status_message.emit(f"Image loaded: {file_path.name} ({image_info.width}x{image_info.height})")
        logger.info(f"Image {file_path.name} loaded successfully.")

    def display_image(self) -> None:
        """
        Converts the currently loaded OpenCV image data to a QPixmap and emits it for display.
        """
        if self._current_image_data is None:
            self.error_occurred.emit("No image data to display.")
            return

        h, w, ch = self._current_image_data.shape
        bytes_per_line = ch * w
        
        # OpenCV uses BGR, QImage expects RGB or BGR depending on format
        # Convert BGR to RGB for QImage
        image_rgb = cv2.cvtColor(self._current_image_data, cv2.COLOR_BGR2RGB)
        
        q_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        if self._current_image_info:
            self.image_loaded.emit(pixmap, self._current_image_info)
        else:
            # Fallback if image_info somehow got lost
            self.image_loaded.emit(pixmap, ImageInfo(Path("unknown.jpg"), w, h, "UNKNOWN"))

    def clear_image(self) -> None:
        """
        Clears the currently loaded image from the controller and notifies the UI.
        """
        self._current_image_data = None
        self._current_image_info = None
        self.image_cleared.emit()
        self.status_message.emit("Image cleared.")
        logger.info("Current image cleared from controller.")
