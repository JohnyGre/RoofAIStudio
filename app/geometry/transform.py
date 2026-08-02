"""
This module provides services for applying geometric transformations to 2D and 3D points and objects.
"""

from typing import Union, List
import numpy as np

from app.geometry.point import Point2D, Point3D, Point
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D

class GeometryTransform:
    """
    Service for applying common geometric transformations (translate, rotate, scale, mirror).
    """

    @staticmethod
    def translate(obj: Union[Point, Edge, Polygon2D, List[Point]], dx: float, dy: float, dz: float = 0.0) -> Union[Point, Edge, Polygon2D, List[Point]]:
        """
        Translates a geometric object by (dx, dy, dz).

        Args:
            obj (Union[Point, Edge, Polygon2D, List[Point]]): The object(s) to translate.
            dx (float): Translation along the x-axis.
            dy (float): Translation along the y-axis.
            dz (float): Translation along the z-axis (only applies to Point3D).

        Returns:
            Union[Point, Edge, Polygon2D, List[Point]]: The translated object(s).
        """
        if isinstance(obj, Point2D):
            return Point2D(obj.x + dx, obj.y + dy)
        elif isinstance(obj, Point3D):
            return Point3D(obj.x + dx, obj.y + dy, obj.z + dz)
        elif isinstance(obj, Edge):
            return Edge(
                GeometryTransform.translate(obj.start_point, dx, dy, dz),
                GeometryTransform.translate(obj.end_point, dx, dy, dz)
            )
        elif isinstance(obj, Polygon2D):
            translated_vertices = [GeometryTransform.translate(v, dx, dy) for v in obj.vertices]
            return Polygon2D(translated_vertices)
        elif isinstance(obj, list) and all(isinstance(p, Point) for p in obj):
            return [GeometryTransform.translate(p, dx, dy, dz) for p in obj]
        else:
            raise TypeError(f"Unsupported object type for translation: {type(obj)}")

    @staticmethod
    def rotate_2d(obj: Union[Point2D, Edge, Polygon2D, List[Point2D]], angle_degrees: float, center: Point2D = Point2D(0.0, 0.0)) -> Union[Point2D, Edge, Polygon2D, List[Point2D]]:
        """
        Rotates a 2D geometric object around a specified center point.

        Args:
            obj (Union[Point2D, Edge, Polygon2D, List[Point2D]]): The 2D object(s) to rotate.
            angle_degrees (float): Rotation angle in degrees (positive for counter-clockwise).
            center (Point2D): The center of rotation.

        Returns:
            Union[Point2D, Edge, Polygon2D, List[Point2D]]: The rotated object(s).
        """
        angle_rad = np.radians(angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        def _rotate_point_2d(p: Point2D) -> Point2D:
            # Translate point so center is at origin
            temp_x = p.x - center.x
            temp_y = p.y - center.y

            # Rotate point
            rotated_x = temp_x * cos_angle - temp_y * sin_angle
            rotated_y = temp_x * sin_angle + temp_y * cos_angle

            # Translate point back
            return Point2D(rotated_x + center.x, rotated_y + center.y)

        if isinstance(obj, Point2D):
            return _rotate_point_2d(obj)
        elif isinstance(obj, Edge):
            return Edge(
                _rotate_point_2d(obj.start_point),
                _rotate_point_2d(obj.end_point)
            )
        elif isinstance(obj, Polygon2D):
            rotated_vertices = [_rotate_point_2d(v) for v in obj.vertices]
            return Polygon2D(rotated_vertices)
        elif isinstance(obj, list) and all(isinstance(p, Point2D) for p in obj):
            return [_rotate_point_2d(p) for p in obj]
        else:
            raise TypeError(f"Unsupported 2D object type for rotation: {type(obj)}")

    @staticmethod
    def scale(obj: Union[Point, Edge, Polygon2D, List[Point]], sx: float, sy: float, sz: float = 1.0, center: Optional[Point] = None) -> Union[Point, Edge, Polygon2D, List[Point]]:
        """
        Scales a geometric object by factors (sx, sy, sz) relative to an optional center.

        Args:
            obj (Union[Point, Edge, Polygon2D, List[Point]]): The object(s) to scale.
            sx (float): Scaling factor along the x-axis.
            sy (float): Scaling factor along the y-axis.
            sz (float): Scaling factor along the z-axis (only applies to Point3D).
            center (Optional[Point]): The center of scaling. If None, scales relative to origin (0,0,0).

        Returns:
            Union[Point, Edge, Polygon2D, List[Point]]: The scaled object(s).
        """
        if center is None:
            if isinstance(obj, Point3D) or (isinstance(obj, list) and obj and isinstance(obj[0], Point3D)):
                center = Point3D(0.0, 0.0, 0.0)
            else:
                center = Point2D(0.0, 0.0)

        def _scale_point(p: Point) -> Point:
            if isinstance(p, Point2D) and isinstance(center, Point2D):
                return Point2D(
                    center.x + (p.x - center.x) * sx,
                    center.y + (p.y - center.y) * sy
                )
            elif isinstance(p, Point3D) and isinstance(center, Point3D):
                return Point3D(
                    center.x + (p.x - center.x) * sx,
                    center.y + (p.y - center.y) * sy,
                    center.z + (p.z - center.z) * sz
                )
            else:
                raise TypeError("Inconsistent point and center types for scaling.")

        if isinstance(obj, Point):
            return _scale_point(obj)
        elif isinstance(obj, Edge):
            return Edge(
                _scale_point(obj.start_point),
                _scale_point(obj.end_point)
            )
        elif isinstance(obj, Polygon2D):
            scaled_vertices = [_scale_point(v) for v in obj.vertices]
            return Polygon2D(scaled_vertices)
        elif isinstance(obj, list) and all(isinstance(p, Point) for p in obj):
            return [_scale_point(p) for p in obj]
        else:
            raise TypeError(f"Unsupported object type for scaling: {type(obj)}")

    @staticmethod
    def mirror_2d(obj: Union[Point2D, Edge, Polygon2D, List[Point2D]], axis: Literal["x", "y"], mirror_line_coord: float = 0.0) -> Union[Point2D, Edge, Polygon2D, List[Point2D]]:
        """
        Mirrors a 2D geometric object across a specified axis (x or y).

        Args:
            obj (Union[Point2D, Edge, Polygon2D, List[Point2D]]): The 2D object(s) to mirror.
            axis (Literal["x", "y"]): The axis to mirror across.
            mirror_line_coord (float): The coordinate of the mirror line (e.g., if axis='x' and coord=5,
                                       the mirror line is x=5).

        Returns:
            Union[Point2D, Edge, Polygon2D, List[Point2D]]: The mirrored object(s).
        """
        def _mirror_point_2d(p: Point2D) -> Point2D:
            if axis == "x": # Mirror across a vertical line x = mirror_line_coord
                return Point2D(2 * mirror_line_coord - p.x, p.y)
            elif axis == "y": # Mirror across a horizontal line y = mirror_line_coord
                return Point2D(p.x, 2 * mirror_line_coord - p.y)
            else:
                raise ValueError("Axis must be 'x' or 'y'.")

        if isinstance(obj, Point2D):
            return _mirror_point_2d(obj)
        elif isinstance(obj, Edge):
            return Edge(
                _mirror_point_2d(obj.start_point),
                _mirror_point_2d(obj.end_point)
            )
        elif isinstance(obj, Polygon2D):
            mirrored_vertices = [_mirror_point_2d(v) for v in obj.vertices]
            return Polygon2D(mirrored_vertices)
        elif isinstance(obj, list) and all(isinstance(p, Point2D) for p in obj):
            return [_mirror_point_2d(p) for p in obj]
        else:
            raise TypeError(f"Unsupported 2D object type for mirroring: {type(obj)}")
