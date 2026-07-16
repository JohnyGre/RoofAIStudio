"""
Unit tests for RoofPostProcessor.
Covers: merging overlapping polygons, extracting shared boundary edges,
and structuring features with parent plane associations.
"""

import pytest
import numpy as np
import uuid

from app.ai.ai_result import DetectionResult, PolygonGeometry, BBoxGeometry
from app.ai.post_processing.roof_post_processor import RoofPostProcessor, RoofFeature
from app.geometry.point import Point2D

def test_merge_overlapping_planes():
    # Define two highly overlapping squares
    # Square 1: (0,0) to (100,100)
    p1_verts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    # Square 2: (5,5) to (105,105) - 90% overlap
    p2_verts = [(5.0, 5.0), (105.0, 5.0), (105.0, 105.0), (5.0, 105.0)]

    dr1 = DetectionResult(
        class_name="roof_plane",
        geometry=PolygonGeometry(p1_verts),
        confidence=0.9
    )
    dr2 = DetectionResult(
        class_name="roof_plane",
        geometry=PolygonGeometry(p2_verts),
        confidence=0.85
    )

    processor = RoofPostProcessor(overlap_merge_threshold=0.8)
    results = processor.post_process([dr1, dr2])

    # Should merge them and keep the higher confidence one (dr1)
    assert len(results.planes) == 1
    assert results.planes[0].id == dr1.id

def test_extract_shared_edges():
    # Define two adjacent squares sharing a boundary from x=100, y=0 to y=100
    # Square 1: (0,0) to (100,100)
    p1_verts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    # Square 2: (100,0) to (200,100)
    p2_verts = [(100.0, 0.0), (200.0, 0.0), (200.0, 100.0), (100.0, 100.0)]

    dr1 = DetectionResult(
        class_name="roof_plane",
        geometry=PolygonGeometry(p1_verts),
        confidence=0.9
    )
    dr2 = DetectionResult(
        class_name="roof_plane",
        geometry=PolygonGeometry(p2_verts),
        confidence=0.9
    )

    processor = RoofPostProcessor(adjacency_threshold_pixels=5.0)
    results = processor.post_process([dr1, dr2])

    # Should find one shared boundary edge
    assert len(results.edges) == 1
    edge = results.edges[0]
    assert edge.left_plane_id in (str(dr1.id), str(dr2.id))
    assert edge.right_plane_id in (str(dr1.id), str(dr2.id))
    assert edge.left_plane_id != edge.right_plane_id
    
    # Check start and end points of the shared edge
    # Start/end should correspond to the shared segment (100, 0) and (100, 100)
    xs = [edge.start_point.x, edge.end_point.x]
    ys = [edge.start_point.y, edge.end_point.y]
    assert 100.0 in xs
    assert 0.0 in ys or 100.0 in ys

def test_structure_features_and_associate_parent():
    # Plane: (0,0) to (100,100)
    plane_verts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    dr_plane = DetectionResult(
        class_name="roof_plane",
        geometry=PolygonGeometry(plane_verts),
        confidence=0.95
    )

    # Opening inside the plane: (30,30) to (50,50)
    dr_opening = DetectionResult(
        class_name="opening",
        geometry=BBoxGeometry(x_min=30.0, y_min=30.0, x_max=50.0, y_max=50.0),
        confidence=0.88
    )

    processor = RoofPostProcessor()
    results = processor.post_process([dr_plane, dr_opening])

    # Should structure the opening and associate it with the parent plane
    assert len(results.features) == 1
    feature = results.features[0]
    assert feature.feature_type == "opening"
    assert feature.parent_plane_id == dr_plane.id
