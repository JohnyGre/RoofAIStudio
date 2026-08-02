"""
This module defines 2D and 3D point data structures and related operations.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Union

@dataclass(frozen=True)
class Point2D:
    """
    Represents a 2D point with x and y coordinates.
    Immutable for safe use in collections and as hashable keys.
    """
    x: float
    y: float

    def distance_to(self, other: "Point2D") -> float:
        """
        Calculates the Euclidean distance to another 2D point.

        Args:
            other (Point2D): The other 2D point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __add__(self, other: "Point2D") -> "Point2D":
        """Adds two Point2D objects."""
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point2D") -> "Point2D":
        """Subtracts two Point2D objects."""
        return Point2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Point2D":
        """Multiplies a Point2D by a scalar."""
        return Point2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Point2D":
        """Divides a Point2D by a scalar."""
        if scalar == 0:
            raise ValueError("Cannot divide by zero.")
        return Point2D(self.x / scalar, self.y / scalar)

@dataclass(frozen=True)
class Point3D:
    """
    Represents a 3D point with x, y, and z coordinates.
    Immutable for safe use in collections and as hashable keys.
    """
    x: float
    y: float
    z: float

    def distance_to(self, other: "Point3D") -> float:
        """
        Calculates the Euclidean distance to another 3D point.

        Args:
            other (Point3D): The other 3D point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def __add__(self, other: "Point3D") -> "Point3D":
        """Adds two Point3D objects."""
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        """Subtracts two Point3D objects."""
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Point3D":
        """Multiplies a Point3D by a scalar."""
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> "Point3D":
        """Divides a Point3D by a scalar."""
        if scalar == 0:
            raise ValueError("Cannot divide by zero.")
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

# Type alias for convenience
Point = Union[Point2D, Point3D]
