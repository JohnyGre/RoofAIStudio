"""
This module defines the Edge data structure for geometric representations.
"""

from dataclasses import dataclass
from typing import Tuple

from app.geometry.point import Point2D, Point3D, Point

@dataclass(frozen=True)
class Edge:
    """
    Represents a directed edge connecting two points.
    Immutable for safe use.
    """
    start_point: Point
    end_point: Point

    @property
    def length(self) -> float:
        """
        Calculates the Euclidean length of the edge.
        """
        return self.start_point.distance_to(self.end_point)

    @property
    def direction(self) -> Point:
        """
        Calculates the normalized direction vector of the edge.
        Returns a Point2D or Point3D depending on the input points.
        """
        delta = self.end_point - self.start_point
        len_val = self.length
        if len_val == 0:
            if isinstance(self.start_point, Point3D):
                return Point3D(0.0, 0.0, 0.0)
            return Point2D(0.0, 0.0)
        return delta / len_val

    def __post_init__(self):
        """
        Ensures that start_point and end_point are of the same type (2D or 3D).
        """
        if type(self.start_point) is not type(self.end_point):
            raise TypeError("Start and end points of an Edge must be of the same type (Point2D or Point3D).")
