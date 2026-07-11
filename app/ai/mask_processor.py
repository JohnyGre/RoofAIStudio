"""
This module provides utilities for processing segmentation masks.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2

from app.geometry.point import Point2D
from app.geometry.polygon import Polygon2D
from app.core.logger import setup_logging

logger = setup_logging()

class MaskProcessor:
    """
    Provides static methods for manipulating and extracting information from segmentation masks.
    """

    @staticmethod
    def clean_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
        """
        Cleans a binary mask using morphological operations (opening and closing).

        Args:
            mask (np.ndarray): The input binary mask (0s and 1s).
            kernel_size (int): Size of the kernel for morphological operations.
            iterations (int): Number of iterations for morphological operations.

        Returns:
            np.ndarray: The cleaned binary mask.
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Opening: erosion followed by dilation (removes small objects)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        # Closing: dilation followed by erosion (fills small holes)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        return cleaned

    @staticmethod
    def remove_noise(mask: np.ndarray, min_area_threshold: int = 100) -> np.ndarray:
        """
        Removes small noisy regions from a binary mask based on contour area.

        Args:
            mask (np.ndarray): The input binary mask (0s and 1s).
            min_area_threshold (int): Minimum contour area to keep.

        Returns:
            np.ndarray: Mask with small noisy regions removed.
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) > min_area_threshold:
                cv2.drawContours(cleaned_mask, [contour], -1, 255, cv2.FILLED)
        return cleaned_mask

    @staticmethod
    def extract_contours(mask: np.ndarray, min_area_threshold: int = 0) -> List[np.ndarray]:
        """
        Extracts contours from a binary mask.

        Args:
            mask (np.ndarray): The input binary mask (0s and 1s).
            min_area_threshold (int): Minimum contour area to return.

        Returns:
            List[np.ndarray]: A list of contours (each a NumPy array of points).
        """
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area_threshold]
        return filtered_contours

    @staticmethod
    def simplify_polygon(contour: np.ndarray, epsilon_factor: float = 0.005) -> Polygon2D:
        """
        Simplifies a contour (polygon) using the Ramer-Douglas-Peucker algorithm.

        Args:
            contour (np.ndarray): A contour (NumPy array of points).
            epsilon_factor (float): Parameter specifying the approximation accuracy.
                                    It is multiplied by the contour's perimeter.

        Returns:
            Polygon2D: A simplified Polygon2D object.
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

        # Convert approximated polygon points to Point2D
        vertices = [Point2D(float(p[0][0]), float(p[0][1])) for p in approx_polygon]
        
        if len(vertices) < 3:
            # If simplification results in less than 3 vertices, return the original contour
            # or handle as an error/degenerate case. For now, use original if too few.
            logger.warning("Polygon simplification resulted in less than 3 vertices. Using original contour.")
            vertices = [Point2D(float(p[0][0]), float(p[0][1])) for p in contour]
            if len(vertices) < 3:
                raise ValueError("Contour has less than 3 vertices even before simplification.")

        return Polygon2D(vertices=vertices)
