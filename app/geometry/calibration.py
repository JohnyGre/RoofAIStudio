"""
This module provides services for calibrating image pixels to real-world measurements.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from app.geometry.point import Point2D

@dataclass(frozen=True)
class CalibrationModel:
    """
    Represents a 2D image calibration model.
    """
    reference_points_pixel: tuple[Point2D, Point2D]  # Two points in pixel coordinates
    reference_distance_meters: float                 # Real-world distance between reference points in meters
    scale_factor_pixels_per_meter: float             # Pixels per meter
    unit: Literal["mm", "cm", "m"]                   # Unit of measurement for convenience

    def __post_init__(self):
        if self.reference_distance_meters <= 0:
            raise ValueError("Reference distance must be positive.")
        if self.scale_factor_pixels_per_meter <= 0:
            raise ValueError("Scale factor must be positive.")

class CalibrationService:
    """
    Service for performing 2D image calibration and unit conversions.
    """

    @staticmethod
    def calibrate_from_distance(
        point1_pixel: Point2D,
        point2_pixel: Point2D,
        real_world_distance_meters: float,
        unit: Literal["mm", "cm", "m"] = "m"
    ) -> CalibrationModel:
        """
        Calibrates the image by calculating the pixel-to-meter scale factor
        based on two pixel points and their known real-world distance.

        Args:
            point1_pixel (Point2D): The first reference point in pixel coordinates.
            point2_pixel (Point2D): The second reference point in pixel coordinates.
            real_world_distance_meters (float): The known real-world distance between
                                                point1 and point2 in meters.
            unit (Literal["mm", "cm", "m"]): The primary unit for this calibration.

        Returns:
            CalibrationModel: An object containing the calibration data.

        Raises:
            ValueError: If real_world_distance_meters is non-positive or points are identical.
        """
        if real_world_distance_meters <= 0:
            raise ValueError("Real-world distance must be positive for calibration.")

        pixel_distance = point1_pixel.distance_to(point2_pixel)
        if pixel_distance == 0:
            raise ValueError("Reference pixel points are identical, cannot calibrate.")

        scale_factor_pixels_per_meter = pixel_distance / real_world_distance_meters

        return CalibrationModel(
            reference_points_pixel=(point1_pixel, point2_pixel),
            reference_distance_meters=real_world_distance_meters,
            scale_factor_pixels_per_meter=scale_factor_pixels_per_meter,
            unit=unit
        )

    @staticmethod
    def pixel_to_meter(pixels: float, calibration: CalibrationModel) -> float:
        """
        Converts a distance in pixels to meters using the provided calibration.

        Args:
            pixels (float): Distance in pixels.
            calibration (CalibrationModel): The calibration model.

        Returns:
            float: The distance in meters.
        """
        return pixels / calibration.scale_factor_pixels_per_meter

    @staticmethod
    def meter_to_pixel(meters: float, calibration: CalibrationModel) -> float:
        """
        Converts a distance in meters to pixels using the provided calibration.

        Args:
            meters (float): Distance in meters.
            calibration (CalibrationModel): The calibration model.

        Returns:
            float: The distance in pixels.
        """
        return meters * calibration.scale_factor_pixels_per_meter

    @staticmethod
    def convert_unit(value: float, from_unit: Literal["mm", "cm", "m"], to_unit: Literal["mm", "cm", "m"]) -> float:
        """
        Converts a value from one real-world unit to another.

        Args:
            value (float): The value to convert.
            from_unit (Literal["mm", "cm", "m"]): The unit of the input value.
            to_unit (Literal["mm", "cm", "m"]): The target unit for conversion.

        Returns:
            float: The converted value.
        """
        conversion_factors = {
            "mm": {"mm": 1, "cm": 0.1, "m": 0.001},
            "cm": {"mm": 10, "cm": 1, "m": 0.01},
            "m": {"mm": 1000, "cm": 100, "m": 1},
        }
        if from_unit not in conversion_factors or to_unit not in conversion_factors[from_unit]:
            raise ValueError(f"Unsupported unit conversion: from {from_unit} to {to_unit}")

        return value * conversion_factors[from_unit][to_unit]
