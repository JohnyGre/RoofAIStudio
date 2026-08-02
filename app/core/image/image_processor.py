"""
This module provides image processing functionalities for Roof AI Studio.
It uses OpenCV and NumPy for efficient image manipulation.
"""

from typing import Tuple, Optional

import cv2
import numpy as np

class ImageProcessor:
    """
    Provides a collection of static methods for common image processing operations.
    All methods operate on NumPy arrays (OpenCV format: BGR).
    """

    @staticmethod
    def resize(image: np.ndarray, new_size: Tuple[int, int], interpolation=cv2.INTER_AREA) -> Optional[np.ndarray]:
        """
        Resizes an image to a specified new size.

        Args:
            image (np.ndarray): The input image (NumPy array, BGR).
            new_size (Tuple[int, int]): The new size as (width, height).
            interpolation: Interpolation method (e.g., cv2.INTER_AREA, cv2.INTER_LINEAR).

        Returns:
            Optional[np.ndarray]: The resized image, or None if input is invalid.
        """
        if image is None or not isinstance(image, np.ndarray):
            print("Error: Invalid image input for resize.")
            return None
        return cv2.resize(image, new_size, interpolation=interpolation)

    @staticmethod
    def rotate(image: np.ndarray, angle: float, scale: float = 1.0) -> Optional[np.ndarray]:
        """
        Rotates an image by a given angle around its center.

        Args:
            image (np.ndarray): The input image (NumPy array, BGR).
            angle (float): The rotation angle in degrees. Positive values mean counter-clockwise rotation.
            scale (float): Scaling factor for the image.

        Returns:
            Optional[np.ndarray]: The rotated image, or None if input is invalid.
        """
        if image is None or not isinstance(image, np.ndarray):
            print("Error: Invalid image input for rotate.")
            return None

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image

    @staticmethod
    def crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        Crops an image to the specified region.

        Args:
            image (np.ndarray): The input image (NumPy array, BGR).
            x (int): X-coordinate of the top-left corner of the crop rectangle.
            y (int): Y-coordinate of the top-left corner of the crop rectangle.
            width (int): Width of the crop rectangle.
            height (int): Height of the crop rectangle.

        Returns:
            Optional[np.ndarray]: The cropped image, or None if input is invalid or crop region is out of bounds.
        """
        if image is None or not isinstance(image, np.ndarray):
            print("Error: Invalid image input for crop.")
            return None

        h, w = image.shape[:2]
        if not (0 <= x < w and 0 <= y < h and x + width <= w and y + height <= h):
            print(f"Error: Crop region ({x},{y},{width},{height}) is out of image bounds ({w},{h}).")
            return None

        cropped_image = image[y:y+height, x:x+width]
        return cropped_image

    @staticmethod
    def normalize(image: np.ndarray, alpha: float = 0, beta: float = 255) -> Optional[np.ndarray]:
        """
        Normalizes the pixel values of an image to a specified range.

        Args:
            image (np.ndarray): The input image (NumPy array, BGR).
            alpha (float): Minimum value in the normalized range.
            beta (float): Maximum value in the normalized range.

        Returns:
            Optional[np.ndarray]: The normalized image, or None if input is invalid.
        """
        if image is None or not isinstance(image, np.ndarray):
            print("Error: Invalid image input for normalize.")
            return None
        normalized_image = cv2.normalize(image, None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX)
        return normalized_image

    @staticmethod
    def convert_color_space(image: np.ndarray, code: int) -> Optional[np.ndarray]:
        """
        Converts the color space of an image.

        Args:
            image (np.ndarray): The input image (NumPy array, BGR).
            code (int): Color conversion code (e.g., cv2.COLOR_BGR2GRAY, cv2.COLOR_BGR2RGB).

        Returns:
            Optional[np.ndarray]: The image in the new color space, or None if input is invalid.
        """
        if image is None or not isinstance(image, np.ndarray):
            print("Error: Invalid image input for convert_color_space.")
            return None
        converted_image = cv2.cvtColor(image, code)
        return converted_image

    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts a BGR image to grayscale.
        """
        return ImageProcessor.convert_color_space(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> Optional[np.ndarray]:
        """
        Converts a BGR image to RGB.
        """
        return ImageProcessor.convert_color_space(image, cv2.COLOR_BGR2RGB)
