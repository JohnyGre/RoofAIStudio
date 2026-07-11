"""
This module provides services for performing geometric measurements on roof components.
"""

from typing import Union, List, Literal

from app.geometry.point import Point2D, Point3D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel, CalibrationService

class MeasurementService:
    """
    Service for calculating various geometric measurements in real-world units.
    """

    @staticmethod
    def calculate_distance(
        point1: Union[Point2D, Point3D],
        point2: Union[Point2D, Point3D],
        calibration: Optional[CalibrationModel] = None,
        output_unit: Literal["mm", "cm", "m"] = "m"
    ) -> float:
        """
        Calculates the real-world distance between two points.
        If a calibration model is provided, it assumes points are in pixels and converts.
        Otherwise, it assumes points are already in real-world units.

        Args:
            point1 (Union[Point2D, Point3D]): The first point.
            point2 (Union[Point2D, Point3D]): The second point.
            calibration (Optional[CalibrationModel]): The calibration model for pixel-to-meter conversion.
                                                      Required if points are in pixels.
            output_unit (Literal["mm", "cm", "m"]): The desired output unit.

        Returns:
            float: The distance in the specified output unit.

        Raises:
            ValueError: If calibration is missing for pixel points, or points are of different types.
        """
        if type(point1) is not type(point2):
            raise ValueError("Points must be of the same type (Point2D or Point3D).")

        distance_in_base_unit: float
        if calibration:
            if not isinstance(point1, Point2D):
                raise ValueError("Calibration is only applicable for Point2D (pixel) measurements.")
            pixel_distance = point1.distance_to(point2)
            distance_in_base_unit = CalibrationService.pixel_to_meter(pixel_distance, calibration)
        else:
            # Assume points are already in meters if no calibration is provided
            distance_in_base_unit = point1.distance_to(point2)

        return CalibrationService.convert_unit(distance_in_base_unit, "m", output_unit)

    @staticmethod
    def calculate_edge_length(
        edge: Edge,
        calibration: Optional[CalibrationModel] = None,
        output_unit: Literal["mm", "cm", "m"] = "m"
    ) -> float:
        """
        Calculates the real-world length of an edge.

        Args:
            edge (Edge): The edge to measure.
            calibration (Optional[CalibrationModel]): The calibration model.
            output_unit (Literal["mm", "cm", "m"]): The desired output unit.

        Returns:
            float: The length in the specified output unit.
        """
        return MeasurementService.calculate_distance(edge.start_point, edge.end_point, calibration, output_unit)

    @staticmethod
    def calculate_area(
        polygon: Polygon2D,
        calibration: Optional[CalibrationModel] = None,
        output_unit: Literal["sq_mm", "sq_cm", "sq_m"] = "sq_m"
    ) -> float:
        """
        Calculates the real-world area of a 2D polygon.
        If a calibration model is provided, it assumes polygon vertices are in pixels and converts.
        Otherwise, it assumes vertices are already in real-world units (meters).

        Args:
            polygon (Polygon2D): The polygon to measure.
            calibration (Optional[CalibrationModel]): The calibration model for pixel-to-meter conversion.
            output_unit (Literal["sq_mm", "sq_cm", "sq_m"]): The desired output unit.

        Returns:
            float: The area in the specified output unit.

        Raises:
            ValueError: If calibration is missing for pixel polygons.
        """
        area_in_sq_meters: float
        if calibration:
            # Area scales by the square of the linear scale factor
            area_in_sq_pixels = polygon.area
            scale_factor_sq_pixels_per_sq_meter = calibration.scale_factor_pixels_per_meter ** 2
            area_in_sq_meters = area_in_sq_pixels / scale_factor_sq_pixels_per_sq_meter
        else:
            # Assume polygon vertices are already in meters
            area_in_sq_meters = polygon.area

        # Convert to desired output unit
        if output_unit == "sq_mm":
            return area_in_sq_meters * (1000 ** 2)
        elif output_unit == "sq_cm":
            return area_in_sq_meters * (100 ** 2)
        elif output_unit == "sq_m":
            return area_in_sq_meters
        else:
            raise ValueError(f"Unsupported output area unit: {output_unit}")

    @staticmethod
    def calculate_perimeter(
        polygon: Polygon2D,
        calibration: Optional[CalibrationModel] = None,
        output_unit: Literal["mm", "cm", "m"] = "m"
    ) -> float:
        """
        Calculates the real-world perimeter of a 2D polygon.

        Args:
            polygon (Polygon2D): The polygon to measure.
            calibration (Optional[CalibrationModel]): The calibration model.
            output_unit (Literal["mm", "cm", "m"]): The desired output unit.

        Returns:
            float: The perimeter in the specified output unit.
        """
        perimeter_in_meters: float
        if calibration:
            perimeter_in_pixels = polygon.perimeter
            perimeter_in_meters = CalibrationService.pixel_to_meter(perimeter_in_pixels, calibration)
        else:
            perimeter_in_meters = polygon.perimeter

        return CalibrationService.convert_unit(perimeter_in_meters, "m", output_unit)

    @staticmethod
    def calculate_roof_geometry_total_area(
        roof_geometry: RoofGeometry,
        calibration: Optional[CalibrationModel] = None,
        output_unit: Literal["sq_mm", "sq_cm", "sq_m"] = "sq_m"
    ) -> float:
        """
        Calculates the total true surface area of a RoofGeometry object.
        Assumes RoofGeometry's internal units are consistent (e.g., meters) if no calibration.
        If calibration is provided, it implies the input geometry might be pixel-based
        and needs conversion, though RoofGeometry itself is designed for real-world units.
        This method primarily uses the RoofGeometry's internal `calculate_total_area`
        and then converts the unit.

        Args:
            roof_geometry (RoofGeometry): The roof geometry model.
            calibration (Optional[CalibrationModel]): Calibration model (if geometry derived from pixels).
            output_unit (Literal["sq_mm", "sq_cm", "sq_m"]): The desired output unit.

        Returns:
            float: The total true surface area in the specified output unit.
        """
        # RoofGeometry's calculate_total_area already returns true area in its internal units (assumed meters)
        area_in_sq_meters = roof_geometry.calculate_total_area()

        # If calibration is provided, it implies the base units might need to be scaled
        # However, RoofGeometry is designed to hold real-world units.
        # This part might need refinement based on how RoofGeometry is constructed from pixel data.
        # For now, we assume RoofGeometry's internal values are already in meters.
        # If it were constructed from pixel-based polygons, then each plane's area calculation
        # would need to use the calibration.
        # For simplicity, we assume the `true_area` property of RoofPlane already accounts for this
        # or that the RoofGeometry is built from already-calibrated data.

        # Convert to desired output unit
        if output_unit == "sq_mm":
            return area_in_sq_meters * (1000 ** 2)
        elif output_unit == "sq_cm":
            return area_in_sq_meters * (100 ** 2)
        elif output_unit == "sq_m":
            return area_in_sq_meters
        else:
            raise ValueError(f"Unsupported output area unit: {output_unit}")
