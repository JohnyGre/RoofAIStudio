"""
This module defines the Polygon2D data structure and related geometric operations.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint

from app.geometry.point import Point2D

@dataclass(frozen=True)
class Polygon2D:
    """
    Represents a 2D polygon defined by an ordered list of Point2D vertices.
    Immutable for safe use.
    """
    vertices: List[Point2D] = field(default_factory=list)

    def __post_init__(self):
        """
        Validates the polygon upon initialization.
        """
        if len(self.vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        # Ensure vertices are immutable if passed as mutable list
        object.__setattr__(self, 'vertices', tuple(self.vertices))

    def _to_shapely_polygon(self) -> ShapelyPolygon:
        """Converts the Polygon2D to a Shapely Polygon object."""
        return ShapelyPolygon([(p.x, p.y) for p in self.vertices])

    @property
    def area(self) -> float:
        """
        Calculates the area of the polygon.
        Uses the Shoelace formula or Shapely for robustness.
        """
        return self._to_shapely_polygon().area

    @property
    def perimeter(self) -> float:
        """
        Calculates the perimeter of the polygon.
        """
        return self._to_shapely_polygon().length

    @property
    def centroid(self) -> Point2D:
        """
        Calculates the centroid (geometric center) of the polygon.
        """
        shapely_centroid = self._to_shapely_polygon().centroid
        return Point2D(shapely_centroid.x, shapely_centroid.y)

    def contains_point(self, point: Point2D) -> bool:
        """
        Checks if the polygon contains a given point.
        """
        return self._to_shapely_polygon().contains(ShapelyPoint(point.x, point.y))

    def to_numpy_array(self) -> np.ndarray:
        """
        Converts the polygon vertices to a NumPy array of shape (N, 2).
        """
        return np.array([[p.x, p.y] for p in self.vertices])
