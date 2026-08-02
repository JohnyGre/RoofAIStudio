"""
Tests for the app.ai.pipeline.roof_plane_detector module.
"""

import pytest
import numpy as np
import cv2
from math import isclose

from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector
from app.ai.ai_result import DetectionResult, BoundingBox
from app.geometry.point import Point2D

class TestRoofPlaneDetector:

    @pytest.fixture
    def detector(self):
        return RoofPlaneDetector()

    @pytest.fixture
    def sample_image_square_roof(self):
        """Creates a simple image with a square 'roof' in the middle."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a white square (roof) on a black background
        cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
        return img

    @pytest.fixture
    def sample_image_l_shape_roof(self):
        """Creates an image with an L-shaped (concave) 'roof'."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw an L-shape
        contour = np.array([
            [50, 50], [150, 50], [150, 100], [100, 100], [100, 150], [50, 150]
        ], dtype=np.int32)
        cv2.fillPoly(img, [contour], (255, 255, 255))
        return img

    def test_detect_square_roof(self, detector: RoofPlaneDetector, sample_image_square_roof: np.ndarray):
        results = detector.detect(sample_image_square_roof)

        assert len(results) == 1
        dr = results[0]
        assert isinstance(dr, DetectionResult)
        assert dr.class_name == "roof_plane"
        assert isclose(dr.confidence, 0.9)
        assert dr.metadata["source"] == "opencv"
        
        # Check bounding box (should be around the 100x100 square)
        assert isclose(dr.bounding_box.x_min, 50.0)
        assert isclose(dr.bounding_box.y_min, 50.0)
        assert isclose(dr.bounding_box.x_max, 150.0)
        assert isclose(dr.bounding_box.y_max, 150.0)

        # Check polygon vertices (should be 4 for a square)
        polygon_vertices = dr.metadata["polygon_vertices"]
        assert len(polygon_vertices) == 4
        # Order might vary, but points should be present
        expected_points = [(50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0)]
        for p in expected_points:
            assert any(isclose(p[0], vp[0]) and isclose(p[1], vp[1]) for vp in polygon_vertices)
        
        # Check centroid
        assert isclose(dr.metadata["centroid"][0], 100.0)
        assert isclose(dr.metadata["centroid"][1], 100.0)

    def test_detect_l_shape_roof_preserves_concavity(self, detector: RoofPlaneDetector, sample_image_l_shape_roof: np.ndarray):
        results = detector.detect(sample_image_l_shape_roof, approx_poly_epsilon_factor=0.01) # Use smaller epsilon for more vertices

        assert len(results) == 1
        dr = results[0]
        assert isinstance(dr, DetectionResult)
        assert dr.class_name == "roof_plane"

        polygon_vertices = dr.metadata["polygon_vertices"]
        # An L-shape has 6 vertices. If convexHull was used, it would be 4.
        assert len(polygon_vertices) == 6

        # Check if the concave vertex is preserved (e.g., (100, 100))
        # This is a conceptual check, actual coordinates might be slightly off due to approxPolyDP
        concave_vertex_approx = (100.0, 100.0)
        assert any(isclose(concave_vertex_approx[0], vp[0], abs_tol=5) and isclose(concave_vertex_approx[1], vp[1], abs_tol=5) for vp in polygon_vertices)

        # Verify that the polygon is indeed concave (check internal angles)
        # This is a more advanced geometric check, for a basic test, checking vertex count is sufficient.
        # For a robust test, one would reconstruct the Polygon2D and check its convexity.
        # For now, the vertex count check is a good indicator.

    def test_detect_no_roof(self, detector: RoofPlaneDetector):
        """Test detection on an empty image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = detector.detect(img)
        assert len(results) == 0

    def test_detect_with_custom_params(self, detector: RoofPlaneDetector, sample_image_square_roof: np.ndarray):
        results = detector.detect(
            sample_image_square_roof,
            canny_threshold1=10,
            canny_threshold2=30,
            min_contour_area_ratio=0.05,
            confidence_score=0.95
        )
        assert len(results) == 1
        assert isclose(results[0].confidence, 0.95)
        assert results[0].metadata["area_pixels"] > (0.05 * 200 * 200) # Check min area filter
