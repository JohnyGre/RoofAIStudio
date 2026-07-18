"""
This module defines the RoofMeasurementService for calculating real-world measurements
of roof polygons.
"""

from typing import List, Optional
import numpy as np

from app.geometry.point import Point2D
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.geometry.roof_area import RoofAreaCalculator, RoofPolygonMetrics
from app.core.logger import setup_logging

logger = setup_logging()

class RoofMeasurementService:
    """
    Service for converting pixel-based roof polygons into real-world measurements
    using a calibration model and calculating their area, perimeter, and centroid.
    """

    def __init__(self, roof_area_calculator: RoofAreaCalculator):
        """
        Initializes the RoofMeasurementService.

        Args:
            roof_area_calculator (RoofAreaCalculator): An instance of RoofAreaCalculator for geometric calculations.
        """
        self._roof_area_calculator = roof_area_calculator
        logger.info("RoofMeasurementService initialized.")

    def measure_from_points(self, pixel_points: List[Point2D], calibration: CalibrationModel) -> RoofPolygonMetrics:
        """
        Measures the area, perimeter, and centroid of a polygon defined by pixel points,
        converting them to real-world units using the provided calibration.

        Args:
            pixel_points (List[Point2D]): A list of Point2D objects representing the polygon vertices in pixel coordinates.
            calibration (CalibrationModel): The calibration model for pixel-to-meter conversion.

        Returns:
            RoofPolygonMetrics: An object containing the calculated area, perimeter, and centroid in real-world units.

        Raises:
            ValueError: If the polygon has fewer than 3 points or if calibration is invalid.
        """
        if len(pixel_points) < 3:
            raise ValueError("Polygón potrebuje aspoň 3 body pre meranie.")
        if calibration.scale_factor_pixels_per_meter <= 0:
            raise ValueError("Kalibrácia má neplatný mierkový faktor (musí byť > 0).")

        # 1. Prevod pixelových bodov na body v reálnych metroch
        real_world_points: List[Point2D] = []
        for p_pixel in pixel_points:
            x_meter = CalibrationService.pixel_to_meter(p_pixel.x, calibration)
            y_meter = CalibrationService.pixel_to_meter(p_pixel.y, calibration)
            real_world_points.append(Point2D(x_meter, y_meter))

        # 2. Výpočet metrík pomocou RoofAreaCalculator
        metrics = self._roof_area_calculator.calculate(real_world_points)
        
        logger.debug(f"Measured polygon: Area={metrics.area_m2} m2, Perimeter={metrics.perimeter_m} m")
        return metrics
