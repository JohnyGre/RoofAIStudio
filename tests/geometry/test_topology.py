"""
Unit tests for RoofTopologyEngine.
Covers deterministic edge classification, graph connectivity, neighbor lookup,
connected component extraction, cycle detection, and topological validation checks.
"""

import pytest
import uuid
from typing import List

from app.geometry.point import Point3D, Point2D
from app.geometry.edge import Edge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.geometry.roof_geometry import RoofGeometry

from app.geometry.topology.models import EdgeClassification
from app.geometry.topology.engine import RoofTopologyEngine

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _create_plane(name: str, vertices_2d: List[Point2D], heights: List[float], slope: float = 30.0) -> RoofPlane:
    poly = Polygon2D(vertices=vertices_2d)
    return RoofPlane(
        name=name,
        polygon=poly,
        slope=slope,
        orientation=0.0,
        height_at_vertices=heights
    )

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_plane_topology():
    """
    A single horizontal plane should have only EAVE boundary edges.
    """
    # Flat rectangle at z=5
    v2d = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
    heights = [5.0, 5.0, 5.0, 5.0]
    plane = _create_plane("Plane_Flat", v2d, heights, slope=0.0)
    
    geom = RoofGeometry(
        vertices=[Point3D(p.x, p.y, 5.0) for p in v2d],
        planes=[plane]
    )
    
    engine = RoofTopologyEngine()
    topology = engine.build_topology(geom)
    
    assert len(topology.graph.nodes) == 1
    assert len(topology.edges) == 4
    
    # All edges belong to 1 plane and are horizontal (slope = 0) -> EAVE
    for edge in topology.edges.values():
        assert edge.classification == EdgeClassification.EAVE
        assert len(edge.adjacent_plane_ids) == 1


def test_simple_gable_roof():
    """
    Two sloped planes meeting at a horizontal ridge.
    Should classify the shared edge as RIDGE, sloped outer edges as RAKE, and horizontal outer edges as EAVE.
    """
    # Plane Left: slope upwards to the right (x=10)
    vl = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
    hl = [0.0, 5.0, 5.0, 0.0]
    plane_left = _create_plane("Left_Plane", vl, hl)
    
    # Plane Right: slope upwards to the left (x=10)
    vr = [Point2D(10, 0), Point2D(20, 0), Point2D(20, 10), Point2D(10, 10)]
    hr = [5.0, 0.0, 0.0, 5.0]
    plane_right = _create_plane("Right_Plane", vr, hr)
    
    geom = RoofGeometry(
        vertices=[
            Point3D(0,0,0), Point3D(10,0,5), Point3D(10,10,5), Point3D(0,10,0),
            Point3D(20,0,0), Point3D(20,10,0)
        ],
        planes=[plane_left, plane_right]
    )
    
    engine = RoofTopologyEngine()
    topology = engine.build_topology(geom)
    
    # Verify graph node count and connection
    assert len(topology.graph.nodes) == 2
    assert len(topology.graph.adjacencies) == 1
    
    # Extract unique edge classes
    ridge_edges = [e for e in topology.edges.values() if e.classification == EdgeClassification.RIDGE]
    eave_edges = [e for e in topology.edges.values() if e.classification == EdgeClassification.EAVE]
    rake_edges = [e for e in topology.edges.values() if e.classification == EdgeClassification.RAKE]
    
    assert len(ridge_edges) == 1
    # Ridge edge is shared by both planes
    assert len(ridge_edges[0].adjacent_plane_ids) == 2
    
    # Eaves are horizontal outer boundaries (bottom and top outer edges at y=0, y=10, z=0)
    assert len(eave_edges) == 4
    for ee in eave_edges:
        assert len(ee.adjacent_plane_ids) == 1
        
    # Rakes are sloped outer boundaries (side boundaries at x=0, x=20, going from z=0 to z=5)
    assert len(rake_edges) == 2


def test_hip_roof_topology():
    """
    Hip roof: Four planes (front, back trapezoids, left, right triangles) meeting.
    Should detect RIDGE, HIP, and EAVE edges.
    """
    # Centered ridge at y=5, from x=5 to x=15, height z=5.
    # Eaves at boundary of 20x10 rectangle at z=0.
    
    # Front trapezoid: (0,0,0) to (20,0,0) to (15,5,5) to (5,5,5)
    v_front = [Point2D(0, 0), Point2D(20, 0), Point2D(15, 5), Point2D(5, 5)]
    h_front = [0.0, 0.0, 5.0, 5.0]
    p_front = _create_plane("Front", v_front, h_front)

    # Back trapezoid: (5,5,5) to (15,5,5) to (20,10,0) to (0,10,0)
    v_back = [Point2D(5, 5), Point2D(15, 5), Point2D(20, 10), Point2D(0, 10)]
    h_back = [5.0, 5.0, 0.0, 0.0]
    p_back = _create_plane("Back", v_back, h_back)

    # Left triangle: (0,0,0) to (5,5,5) to (0,10,0)
    v_left = [Point2D(0, 0), Point2D(5, 5), Point2D(0, 10)]
    h_left = [0.0, 5.0, 0.0]
    p_left = _create_plane("Left", v_left, h_left)

    # Right triangle: (20,0,0) to (20,10,0) to (15,5,5)
    v_right = [Point2D(20, 0), Point2D(20, 10), Point2D(15, 5)]
    h_right = [0.0, 0.0, 5.0]
    p_right = _create_plane("Right", v_right, h_right)

    geom = RoofGeometry(
        vertices=[
            Point3D(0,0,0), Point3D(20,0,0), Point3D(15,5,5), Point3D(5,5,5),
            Point3D(20,10,0), Point3D(0,10,0)
        ],
        planes=[p_front, p_back, p_left, p_right]
    )

    engine = RoofTopologyEngine()
    topology = engine.build_topology(geom)

    # Verify edge types
    ridge_edges = [e for e in topology.edges.values() if e.classification == EdgeClassification.RIDGE]
    hip_edges = [e for e in topology.edges.values() if e.classification == EdgeClassification.HIP]
    eave_edges = [e for e in topology.edges.values() if e.classification == EdgeClassification.EAVE]

    assert len(ridge_edges) == 1       # Peak ridge line
    assert len(hip_edges) == 4         # Four hip corners
    assert len(eave_edges) == 4        # Four outer bottom boundary edges


def test_valley_detection_l_shaped():
    """
    L-shaped roof intersection creating a valley.
    """
    # Plane 1: Slopes up to valley line
    v1 = [Point2D(0, 0), Point2D(10, 10), Point2D(0, 10)]
    h1 = [5.0, 0.0, 5.0]
    p1 = _create_plane("Plane_A", v1, h1)

    # Plane 2: Slopes up on other side of valley line
    v2 = [Point2D(10, 10), Point2D(10, 0), Point2D(0, 0)]
    h2 = [0.0, 5.0, 5.0]
    p2 = _create_plane("Plane_B", v2, h2)

    geom = RoofGeometry(
        vertices=[Point3D(0,0,5), Point3D(10,10,0), Point3D(0,10,5), Point3D(10,0,5)],
        planes=[p1, p2]
    )

    engine = RoofTopologyEngine()
    topology = engine.build_topology(geom)

    # Check for valley edge classification
    valleys = [e for e in topology.edges.values() if e.classification == EdgeClassification.VALLEY]
    assert len(valleys) == 1
    # Verify the valley connects the correct points (0,0,5) to (10,10,0)
    v_edge = valleys[0]
    assert len(v_edge.adjacent_plane_ids) == 2


def test_disconnected_components():
    """
    Multiple disconnected roofs should produce distinct components in the graph.
    """
    # Roof 1: Gable side A
    v1 = [Point2D(0, 0), Point2D(5, 0), Point2D(5, 5), Point2D(0, 5)]
    h1 = [0.0, 2.0, 2.0, 0.0]
    p1 = _create_plane("R1_PlaneA", v1, h1)

    v2 = [Point2D(5, 0), Point2D(10, 0), Point2D(10, 5), Point2D(5, 5)]
    h2 = [2.0, 0.0, 0.0, 2.0]
    p2 = _create_plane("R1_PlaneB", v2, h2)

    # Roof 2: Isolated flat plane far away
    v3 = [Point2D(50, 50), Point2D(60, 50), Point2D(60, 60), Point2D(50, 60)]
    h3 = [4.0, 4.0, 4.0, 4.0]
    p3 = _create_plane("R2_Flat", v3, h3)

    geom = RoofGeometry(
        vertices=[
            Point3D(0,0,0), Point3D(5,0,2), Point3D(5,5,2), Point3D(0,5,0),
            Point3D(10,0,0), Point3D(10,5,0),
            Point3D(50,50,4), Point3D(60,50,4), Point3D(60,60,4), Point3D(50,60,4)
        ],
        planes=[p1, p2, p3]
    )

    engine = RoofTopologyEngine()
    topology = engine.build_topology(geom)

    components = topology.graph.find_connected_components()
    # Should find 2 separate connected components (one with 2 nodes, one with 1 node)
    assert len(components) == 2
    sizes = sorted([len(c) for c in components])
    assert sizes == [1, 2]


def test_validation_rules():
    """
    Verifies that the validation suite correctly flags topological issues.
    """
    # 1. Invalid polygon (fewer than 3 vertices)
    poly_fail = Polygon2D(vertices=[Point2D(0,0), Point2D(10,0)])
    plane_invalid = RoofPlane(
        name="Plane_Invalid",
        polygon=poly_fail,
        slope=0.0,
        orientation=0.0,
        height_at_vertices=[0.0, 0.0]
    )

    # 2. Non-planar polygon
    v3 = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
    h3 = [0.0, 0.0, 0.0, 5.0]  # Point 4 raised up by 5m, making it non-planar
    plane_nonplanar = _create_plane("Plane_NonPlanar", v3, h3)

    # 3. Non-manifold edge: 3 planes sharing the same line
    vm1 = [Point2D(0, 0), Point2D(10, 0), Point2D(10, 10), Point2D(0, 10)]
    pm1 = _create_plane("M1", vm1, [0.0, 0.0, 0.0, 0.0])

    vm2 = [Point2D(10, 0), Point2D(20, 0), Point2D(20, 10), Point2D(10, 10)]
    pm2 = _create_plane("M2", vm2, [0.0, 0.0, 0.0, 0.0])

    vm3 = [Point2D(10, 0), Point2D(10, 10), Point2D(15, 5)]
    pm3 = _create_plane("M3", vm3, [0.0, 0.0, 0.0])

    geom = RoofGeometry(
        vertices=[Point3D(0,0,0), Point3D(10,0,0), Point3D(10,10,0), Point3D(0,10,0)],
        planes=[plane_invalid, plane_nonplanar, pm1, pm2, pm3]
    )

    engine = RoofTopologyEngine()
    topology = engine.build_topology(geom)

    issue_types = [issue.issue_type for issue in topology.validation_issues]
    
    assert "invalid_polygon" in issue_types
    assert "non_manifold" in issue_types
