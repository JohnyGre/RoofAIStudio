"""
This module provides a service for calculating real-world measurements of roof geometry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional

from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.geometry.point import Point2D, Point3D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.core.logger import setup_logging

logger = setup_logging()

@dataclass(frozen=True)
class RoofMeasurementResult:
    """
    Data model for storing the real-world measurements of a roof geometry.
    """
    total_area_m2: float = 0.0
    total_perimeter_m: float = 0.0
    plane_areas_m2: Dict[str, float] = field(default_factory=dict) # Plane name to area
    edge_lengths_m: Dict[str, float] = field(default_factory=dict) # Edge ID/name to length
    # Add other statistics as needed, e.g., ridge lengths, valley lengths, etc.

class RoofMeasurementService:
    """
    Service for converting pixel-based roof geometry into real-world measurements
    using a calibration model.
    """

    def __init__(self):
        pass

    def calculate_real_area(
        self,
        polygon: Polygon2D,
        calibration: CalibrationModel,
        output_unit: Literal["sq_mm", "sq_cm", "sq_m"] = "sq_m"
    ) -> float:
        """
        Calculates the real-world area of a 2D polygon.
        Assumes the input polygon's vertices are in pixel coordinates.

        Args:
            polygon (Polygon2D): The polygon whose vertices are in pixel coordinates.
            calibration (CalibrationModel): The calibration model for pixel-to-real-world conversion.
            output_unit (Literal["sq_mm", "sq_cm", "sq_m"]): The desired output unit.

        Returns:
            float: The area in the specified output unit.

        Raises:
            ValueError: If calibration is invalid.
        """
        if calibration.scale_factor_pixels_per_meter <= 0:
            raise ValueError("Calibration scale factor must be positive.")

        # Calculate area in square pixels
        area_sq_pixels = polygon.area

        # Convert to square meters
        area_sq_meters = area_sq_pixels / (calibration.scale_factor_pixels_per_meter ** 2)

        # Convert to desired output unit
        if output_unit == "sq_mm":
            return area_sq_meters * (1000 ** 2)
        elif output_unit == "sq_cm":
            return area_sq_meters * (100 ** 2)
        elif output_unit == "sq_m":
            return area_sq_meters
        else:
            raise ValueError(f"Unsupported output area unit: {output_unit}")

    def calculate_real_length(
        self,
        point1: Point2D,
        point2: Point2D,
        calibration: CalibrationModel,
        output_unit: Literal["mm", "cm", "m"] = "m"
    ) -> float:
        """
        Calculates the real-world length between two 2D points.
        Assumes the input points are in pixel coordinates.

        Args:
            point1 (Point2D): The first point in pixel coordinates.
            point2 (Point2D): The second point in pixel coordinates.
            calibration (CalibrationModel): The calibration model for pixel-to-real-world conversion.
            output_unit (Literal["mm", "cm", "m"]): The desired output unit.

        Returns:
            float: The length in the specified output unit.

        Raises:
            ValueError: If calibration is invalid.
        """
        if calibration.scale_factor_pixels_per_meter <= 0:
            raise ValueError("Calibration scale factor must be positive.")

        pixel_distance = point1.distance_to(point2)
        distance_meters = CalibrationService.pixel_to_meter(pixel_distance, calibration)

        return CalibrationService.convert_unit(distance_meters, "m", output_unit)

    def calculate_roof_statistics(
        self,
        roof_geometry: RoofGeometry,
        calibration: Optional[CalibrationModel] = None, # Optional if RoofGeometry already in real-world units
        output_length_unit: Literal["mm", "cm", "m"] = "m",
        output_area_unit: Literal["sq_mm", "sq_cm", "sq_m"] = "sq_m"
    ) -> RoofMeasurementResult:
        """
        Calculates comprehensive real-world measurements for a given RoofGeometry.

        Args:
            roof_geometry (RoofGeometry): The RoofGeometry object.
            calibration (Optional[CalibrationModel]): Calibration model if the RoofGeometry's
                                                      internal units are pixel-based.
                                                      If None, assumes RoofGeometry is already in meters.
            output_length_unit (Literal["mm", "cm", "m"]): Desired unit for lengths.
            output_area_unit (Literal["sq_mm", "sq_cm", "sq_m"]): Desired unit for areas.

        Returns:
            RoofMeasurementResult: An object containing all calculated measurements.

        Raises:
            ValueError: If calibration is required but not provided or invalid.
        """
        if calibration is None:
            logger.warning("No calibration provided. Assuming RoofGeometry is already in real-world units (meters).")
            # If no calibration, we assume the RoofGeometry's internal units are meters.
            # The RoofGeometry's `true_area` property already accounts for slope.
            total_area_m2 = roof_geometry.calculate_total_area()
        else:
            # If calibration is provided, it implies the RoofGeometry might have been
            # constructed from pixel data, and its internal 2D polygons need scaling.
            # This is a more complex scenario. For simplicity, we'll assume
            # that if calibration is provided, the RoofGeometry's 2D polygons
            # are pixel-based, and we need to re-calculate areas.
            # However, the current RoofGeometry structure expects real-world units.
            # A more robust approach would be to have a pixel-based intermediate geometry
            # or ensure RoofGeometry is always built with real-world units.

            # For now, let's assume RoofGeometry's internal polygons are already in meters
            # if calibration is provided, and we just need to convert the final area unit.
            # This means the `calculate_total_area` method of RoofGeometry should
            # internally handle the pixel-to-meter conversion if its planes' polygons
            # were pixel-based.
            # Given the current `RoofPlane` and `RoofGeometry` design, they expect
            # their `polygon` to be in real-world units (meters) if `true_area` is called.
            # So, if calibration is present, it means the geometry was already converted
            # from pixels to meters during its creation (e.g., by GeometryConverter).
            total_area_m2 = roof_geometry.calculate_total_area()


        plane_areas_m2: Dict[str, float] = {}
        for plane in roof_geometry.planes:
            # RoofPlane.true_area already returns area in the unit of its polygon (assumed meters)
            plane_areas_m2[plane.name] = plane.true_area

        # Calculate total perimeter (sum of all unique exterior edges)
        # This requires more sophisticated topological analysis of the RoofGeometry
        # For now, we'll sum all edge lengths as a placeholder.
        total_perimeter_m = sum(edge.length for edge in roof_geometry.edges) # Assuming edges are in meters

        edge_lengths_m: Dict[str, float] = {}
        for i, edge in enumerate(roof_geometry.edges):
            edge_lengths_m[f"Edge_{i}"] = edge.length # Assuming edge.length is in meters

        # Convert total area to desired unit
        final_total_area = self.calculate_real_area(
            Polygon2D(vertices=[Point2D(0,0), Point2D(total_area_m2**0.5, 0), Point2D(total_area_m2**0.5, total_area_m2**0.5), Point2D(0, total_area_m2**0.5)]),
            CalibrationModel(reference_points_pixel=(Point2D(0,0), Point2D(1,0)), reference_distance_meters=1.0, scale_factor_pixels_per_meter=1.0, unit="m"),
            output_area_unit
        )
        
        # Convert plane areas to desired unit
        final_plane_areas: Dict[str, float] = {}
        for name, area_m2 in plane_areas_m2.items():
            final_plane_areas[name] = self.calculate_real_area(
                Polygon2D(vertices=[Point2D(0,0), Point2D(area_m2**0.5, 0), Point2D(area_m2**0.5, area_m2**0.5), Point2D(0, area_m2**0.5)]),
                CalibrationModel(reference_points_pixel=(Point2D(0,0), Point2D(1,0)), reference_distance_meters=1.0, scale_factor_pixels_per_meter=1.0, unit="m"),
                output_area_unit
            )

        # Convert total perimeter and edge lengths to desired unit
        final_total_perimeter = CalibrationService.convert_unit(total_perimeter_m, "m", output_length_unit)
        final_edge_lengths: Dict[str, float] = {}
        for name, length_m in edge_lengths_m.items():
            final_edge_lengths[name] = CalibrationService.convert_unit(length_m, "m", output_length_unit)


        return RoofMeasurementResult(
            total_area_m2=final_total_area,
            total_perimeter_m=final_total_perimeter,
            plane_areas_m2=final_plane_areas,
            edge_lengths_m=final_edge_lengths
        )
