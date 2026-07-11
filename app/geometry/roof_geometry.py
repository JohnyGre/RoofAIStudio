"""
This module defines the comprehensive RoofGeometry data structure.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
import uuid

from app.geometry.point import Point3D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane

@dataclass(frozen=True)
class RoofGeometry:
    """
    Represents the complete 3D geometric model of a roof.
    Immutable for safe use.
    """
    # Unique identifier for this geometry instance
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    # Core geometric components
    vertices: List[Point3D] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    planes: List[RoofPlane] = field(default_factory=list)

    # Specific roof features
    ridges: List[Edge] = field(default_factory=list)
    valleys: List[Edge] = field(default_factory=list)
    openings: List[Polygon2D] = field(default_factory=list) # e.g., skylights, chimneys

    def __post_init__(self):
        """
        Performs initial validation and ensures immutability of lists.
        """
        object.__setattr__(self, 'vertices', tuple(self.vertices))
        object.__setattr__(self, 'edges', tuple(self.edges))
        object.__setattr__(self, 'planes', tuple(self.planes))
        object.__setattr__(self, 'ridges', tuple(self.ridges))
        object.__setattr__(self, 'valleys', tuple(self.valleys))
        object.__setattr__(self, 'openings', tuple(self.openings))

        # Basic validation on creation
        self.validate_geometry()

    def calculate_total_area(self) -> float:
        """
        Calculates the total true surface area of all roof planes,
        subtracting the area of any openings.
        """
        total_plane_area = sum(plane.true_area for plane in self.planes)
        total_opening_area = sum(opening.area for opening in self.openings)
        return max(0.0, total_plane_area - total_opening_area)

    def validate_geometry(self) -> None:
        """
        Performs a series of checks to ensure the geometric consistency of the roof model.
        Raises ValueError if inconsistencies are found.
        """
        # 1. Check for minimum components
        if not self.planes:
            raise ValueError("RoofGeometry must contain at least one roof plane.")

        # 2. Check vertex references in edges and planes
        all_defined_points: Set[Point3D] = set(self.vertices)
        if len(all_defined_points) != len(self.vertices):
            raise ValueError("Duplicate Point3D objects found in vertices list.")

        for edge in self.edges:
            if edge.start_point not in all_defined_points or edge.end_point not in all_defined_points:
                raise ValueError(f"Edge {edge} references undefined vertex.")

        for plane in self.planes:
            for vertex_2d in plane.polygon.vertices:
                # This check is more complex as plane.polygon.vertices are 2D,
                # but they should correspond to the 2D projection of a 3D vertex.
                # For now, we'll assume a mapping mechanism or that 2D points are derived.
                # A more robust check would involve comparing 2D projections of 3D vertices.
                pass # Placeholder for more advanced 2D-3D vertex mapping validation

        # 3. Check for consistent point types (all 3D for RoofGeometry)
        for p in self.vertices:
            if not isinstance(p, Point3D):
                raise TypeError(f"All vertices in RoofGeometry must be Point3D, found {type(p)}")
        for edge in self.edges + self.ridges + self.valleys:
            if not isinstance(edge.start_point, Point3D) or not isinstance(edge.end_point, Point3D):
                raise TypeError(f"All edge points in RoofGeometry must be Point3D, found {type(edge.start_point)} or {type(edge.end_point)}")

        # 4. Check for self-intersections (can be complex, might use Shapely for 2D projections)
        # Placeholder for future implementation
        # For example, check if 2D projections of planes overlap unexpectedly

        # 5. Check if ridges and valleys are valid edges
        for ridge in self.ridges:
            if ridge not in self.edges:
                raise ValueError(f"Ridge {ridge} is not defined as a general edge.")
        for valley in self.valleys:
            if valley not in self.edges:
                raise ValueError(f"Valley {valley} is not defined as a general edge.")

        # 6. Check for valid polygons for openings
        for opening in self.openings:
            if len(opening.vertices) < 3:
                raise ValueError(f"Opening polygon {opening} has less than 3 vertices.")
            # Further checks could ensure openings are within planes, don't overlap, etc.

        # Add more sophisticated checks as the geometry model evolves
        print("RoofGeometry validation passed.")
