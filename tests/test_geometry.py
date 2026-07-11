"""
Tests for the app.geometry module.
"""

import pytest
from math import sqrt, isclose

from app.geometry.point import Point2D, Point3D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.geometry.roof_geometry import RoofGeometry

class TestPoint:
    def test_point2d_creation(self):
        p = Point2D(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.__repr__() == "Point2D(x=1.0, y=2.0)"

    def test_point2d_distance(self):
        p1 = Point2D(0, 0)
        p2 = Point2D(3, 4)
        assert isclose(p1.distance_to(p2), 5.0)

    def test_point2d_immutability(self):
        p = Point2D(1, 2)
        with pytest.raises(AttributeError):
            p.x = 3

    def test_point3d_creation(self):
        p = Point3D(1.0, 2.0, 3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0
        assert p.__repr__() == "Point3D(x=1.0, y=2.0, z=3.0)"

    def test_point3d_distance(self):
        p1 = Point3D(0, 0, 0)
        p2 = Point3D(2, 3, 6)
        assert isclose(p1.distance_to(p2), 7.0)

    def test_point3d_immutability(self):
        p = Point3D(1, 2, 3)
        with pytest.raises(AttributeError):
            p.x = 4

class TestEdge:
    def test_edge_creation(self):
        p1 = Point2D(0, 0)
        p2 = Point2D(5, 0)
        edge = Edge(p1, p2)
        assert edge.start_point == p1
        assert edge.end_point == p2
        assert isclose(edge.length, 5.0)
        assert edge.direction == Point2D(1.0, 0.0)

    def test_edge_3d(self):
        p1 = Point3D(0, 0, 0)
        p2 = Point3D(0, 0, 10)
        edge = Edge(p1, p2)
        assert isclose(edge.length, 10.0)
        assert edge.direction == Point3D(0.0, 0.0, 1.0)

    def test_edge_invalid_points(self):
        p2d = Point2D(0, 0)
        p3d = Point3D(0, 0, 0)
        with pytest.raises(TypeError):
            Edge(p2d, p3d)

class TestPolygon2D:
    def test_polygon2d_creation(self):
        vertices = [Point2D(0,0), Point2D(10,0), Point2D(10,10), Point2D(0,10)]
        polygon = Polygon2D(vertices)
        assert len(polygon.vertices) == 4

    def test_polygon2d_invalid_creation(self):
        with pytest.raises(ValueError):
            Polygon2D(vertices=[Point2D(0,0), Point2D(1,1)]) # Less than 3 vertices

    def test_polygon2d_area(self):
        # Square 10x10
        vertices = [Point2D(0,0), Point2D(10,0), Point2D(10,10), Point2D(0,10)]
        polygon = Polygon2D(vertices)
        assert isclose(polygon.area, 100.0)

        # Triangle
        vertices = [Point2D(0,0), Point22D(5,0), Point2D(0,5)]
        polygon = Polygon2D(vertices)
        assert isclose(polygon.area, 12.5)

    def test_polygon2d_perimeter(self):
        # Square 10x10
        vertices = [Point2D(0,0), Point2D(10,0), Point2D(10,10), Point2D(0,10)]
        polygon = Polygon2D(vertices)
        assert isclose(polygon.perimeter, 40.0)

    def test_polygon2d_centroid(self):
        # Square 10x10
        vertices = [Point2D(0,0), Point2D(10,0), Point2D(10,10), Point2D(0,10)]
        polygon = Polygon2D(vertices)
        centroid = polygon.centroid
        assert isclose(centroid.x, 5.0)
        assert isclose(centroid.y, 5.0)

class TestRoofPlane:
    def test_roof_plane_creation(self, sample_polygon2d: Polygon2D):
        plane = RoofPlane(
            name="TestPlane",
            polygon=sample_polygon2d,
            slope=30.0,
            orientation=45.0,
            height_at_vertices=[0.0, 0.0, 5.0, 5.0]
        )
        assert plane.name == "TestPlane"
        assert plane.polygon == sample_polygon2d
        assert isclose(plane.slope, 30.0)
        assert isclose(plane.orientation, 45.0)
        assert plane.height_at_vertices == [0.0, 0.0, 5.0, 5.0]

    def test_roof_plane_invalid_slope(self, sample_polygon2d: Polygon2D):
        with pytest.raises(ValueError):
            RoofPlane(name="Invalid", polygon=sample_polygon2d, slope=91.0, orientation=0.0)

    def test_roof_plane_area_2d(self, sample_polygon2d: Polygon2D):
        plane = RoofPlane(name="TestPlane", polygon=sample_polygon2d, slope=0.0, orientation=0.0)
        assert isclose(plane.area_2d, 100.0)

    def test_roof_plane_true_area(self, sample_polygon2d: Polygon2D):
        # Flat plane
        plane_flat = RoofPlane(name="Flat", polygon=sample_polygon2d, slope=0.0, orientation=0.0)
        assert isclose(plane_flat.true_area, 100.0)

        # 60 degree slope (cos(60) = 0.5)
        plane_sloped = RoofPlane(name="Sloped", polygon=sample_polygon2d, slope=60.0, orientation=0.0)
        assert isclose(plane_sloped.true_area, 100.0 / 0.5) # 200.0

    def test_roof_plane_get_3d_vertices(self, sample_polygon2d: Polygon2D):
        plane = RoofPlane(
            name="TestPlane",
            polygon=sample_polygon2d,
            slope=30.0,
            orientation=45.0,
            height_at_vertices=[0.0, 0.0, 5.0, 5.0]
        )
        vertices_3d = plane.get_3d_vertices()
        assert len(vertices_3d) == 4
        assert vertices_3d[0] == Point3D(0.0, 0.0, 0.0)
        assert vertices_3d[2] == Point3D(10.0, 10.0, 5.0)

class TestRoofGeometry:
    def test_roof_geometry_creation(self, sample_roof_geometry: RoofGeometry):
        assert len(sample_roof_geometry.vertices) == 4
        assert len(sample_roof_geometry.planes) == 1
        assert sample_roof_geometry.planes[0].name == "Main Plane"

    def test_roof_geometry_validation_minimum_planes(self):
        with pytest.raises(ValueError, match="RoofGeometry must contain at least one roof plane."):
            RoofGeometry(vertices=[], edges=[], planes=[], ridges=[], valleys=[], openings=[])

    def test_roof_geometry_calculate_total_area(self, sample_roof_geometry: RoofGeometry):
        # Assuming sample_roof_geometry has one plane with 2D area 100 and 30 deg slope
        # true_area = 100 / cos(30) = 100 / 0.866 = 115.47
        expected_area = sample_roof_geometry.planes[0].true_area
        assert isclose(sample_roof_geometry.calculate_total_area(), expected_area)

        # Add an opening
        opening_polygon = Polygon2D(vertices=[Point2D(1,1), Point2D(2,1), Point2D(2,2), Point2D(1,2)]) # 1x1 square
        roof_geometry_with_opening = RoofGeometry(
            vertices=sample_roof_geometry.vertices,
            edges=sample_roof_geometry.edges,
            planes=sample_roof_geometry.planes,
            ridges=sample_roof_geometry.ridges,
            valleys=sample_roof_geometry.valleys,
            openings=[opening_polygon]
        )
        # Area of opening is 1.0
        assert isclose(roof_geometry_with_opening.calculate_total_area(), expected_area - 1.0)
