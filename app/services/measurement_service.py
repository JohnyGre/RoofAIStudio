"""
This module provides a service for calculating real-world measurements of roof geometry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional

from app.geometry.roof_geometry import RoofGeometry
from app.geometry.topology.models import RoofTopology
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
        topology: RoofTopology,
        calibration: Optional[CalibrationModel] = None, # Optional if RoofGeometry already in real-world units
        output_length_unit: Literal["mm", "cm", "m"] = "m",
        output_area_unit: Literal["sq_mm", "sq_cm", "sq_m"] = "sq_m"
    ) -> RoofMeasurementResult:
        """
        Calculates comprehensive real-world measurements for a given RoofGeometry.

        Args:
            roof_geometry (RoofGeometry): The RoofGeometry object.
            topology (RoofTopology): The topology corresponding to the roof_geometry.
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

        # Assume roof_geometry is in meters, as per existing comments.
        total_area_m2 = roof_geometry.calculate_total_area()
        
        plane_areas_m2: Dict[str, float] = {plane.name: plane.true_area for plane in roof_geometry.planes}

        # Use topology to get all edge lengths
        edge_lengths_m: Dict[str, float] = {edge.edge_id: edge.length for edge in topology.edges}

        # Calculate true perimeter from outer boundary edges
        total_perimeter_m = sum(edge_lengths_m.get(edge_id, 0) for edge_id in topology.outer_boundary)

        # Convert units for final output
        final_total_area = CalibrationService.convert_area_unit(total_area_m2, "sq_m", output_area_unit)
        
        final_plane_areas: Dict[str, float] = {
            name: CalibrationService.convert_area_unit(area_m2, "sq_m", output_area_unit)
            for name, area_m2 in plane_areas_m2.items()
        }

        final_total_perimeter = CalibrationService.convert_unit(total_perimeter_m, "m", output_length_unit)
        
        final_edge_lengths: Dict[str, float] = {
            name: CalibrationService.convert_unit(length_m, "m", output_length_unit)
            for name, length_m in edge_lengths_m.items()
        }

        return RoofMeasurementResult(
            total_area_m2=final_total_area,
            total_perimeter_m=final_total_perimeter,
            plane_areas_m2=final_plane_areas,
            edge_lengths_m=final_edge_lengths
        )
