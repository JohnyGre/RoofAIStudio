"""
This module defines the RoofPlane data structure for representing individual roof surfaces.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np

from app.geometry.polygon import Polygon2D
from app.geometry.point import Point3D

@dataclass(frozen=True)
class RoofPlane:
    """
    Represents a single plane (face) of a roof, defined by its 2D projection,
    slope, and orientation.
    Immutable for safe use.
    """
    name: str
    polygon: Polygon2D  # 2D projection of the plane on the ground
    slope: float        # Slope of the plane in degrees
    orientation: float  # Orientation (azimuth) of the plane in degrees (e.g., 0=North, 90=East)
    height_at_vertices: List[float] = field(default_factory=list) # Z-coordinates for each polygon vertex
    plane_normal: Optional[Point3D] = None # Normal vector to the plane (for 3D calculations)

    def __post_init__(self):
        """
        Validates the RoofPlane upon initialization.
        """
        if not (0 <= self.slope <= 90):
            raise ValueError("Slope must be between 0 and 90 degrees.")
        if not (0 <= self.orientation < 360):
            raise ValueError("Orientation must be between 0 and 360 degrees.")
        if self.height_at_vertices and len(self.height_at_vertices) != len(self.polygon.vertices):
            raise ValueError("Number of height_at_vertices must match the number of polygon vertices.")

    @property
    def area_2d(self) -> float:
        """
        Returns the 2D projected area of the roof plane.
        """
        return self.polygon.area

    @property
    def true_area(self) -> float:
        """
        Calculates the true surface area of the roof plane, considering its slope.
        Area_true = Area_2D / cos(radians(slope))
        """
        if self.slope == 90: # Vertical plane, infinite true area from 2D projection
            return float('inf')
        if self.slope == 0: # Flat plane
            return self.area_2d
        
        slope_rad = np.radians(self.slope)
        return self.area_2d / np.cos(slope_rad)

    def get_3d_vertices(self) -> List[Point3D]:
        """
        Returns the 3D vertices of the roof plane.
        Assumes height_at_vertices corresponds to polygon.vertices.
        """
        if not self.height_at_vertices:
            raise ValueError("height_at_vertices must be provided to get 3D vertices.")
        return [
            Point3D(v.x, v.y, self.height_at_vertices[i])
            for i, v in enumerate(self.polygon.vertices)
        ]
