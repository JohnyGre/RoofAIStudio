"""
Isolated unit tests for RoofPlaneDetector.
Tests cover: basic detection, multiple polygon detection, metadata structure,
parameter overrides, edge cases, and sorting behaviour.
"""

import pytest
import numpy as np
import cv2

from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector
from app.ai.ai_result import DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blank(h: int = 400, w: int = 400) -> np.ndarray:
    """Return a black BGR image."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_filled_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       color=(255, 255, 255)) -> np.ndarray:
    """Draw a filled rectangle on *img* (in-place) and return it."""
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    return img


def _draw_filled_triangle(img: np.ndarray, pts: list,
                           color=(255, 255, 255)) -> np.ndarray:
    """Draw a filled triangle on *img* (in-place) and return it."""
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoofPlaneDetectorBasic:
    """Basic functionality and contract tests."""

    def test_returns_list(self):
        detector = RoofPlaneDetector()
        img = _make_blank()
        result = detector.detect(img)
        assert isinstance(result, list)

    def test_each_result_is_detection_result(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            assert isinstance(r, DetectionResult)

    def test_class_name_is_roof_plane(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            assert r.class_name == "roof_plane"

    def test_raises_on_none_image(self):
        detector = RoofPlaneDetector()
        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect(None)

    def test_raises_on_non_ndarray_image(self):
        detector = RoofPlaneDetector()
        with pytest.raises(ValueError, match="Invalid image"):
            detector.detect("not_an_image")


class TestRoofPlaneDetectorMetadata:
    """Verify that each detection carries the expected metadata keys."""

    REQUIRED_METADATA_KEYS = {
        "source", "polygon_vertices", "centroid",
        "confidence", "area_pixels", "fill_ratio", "solidity"
    }

    def test_metadata_keys_present(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        assert len(results) >= 1, "Expected at least one detection"
        for r in results:
            missing = self.REQUIRED_METADATA_KEYS - set(r.metadata.keys())
            assert not missing, f"Missing metadata keys: {missing}"

    def test_polygon_vertices_has_at_least_3_points(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            pts = r.metadata["polygon_vertices"]
            assert len(pts) >= 3, f"Polygon has {len(pts)} vertices, need >= 3"

    def test_centroid_inside_image_bounds(self):
        h, w = 400, 400
        img = _make_blank(h, w)
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            cx, cy = r.metadata["centroid"]
            assert 0 <= cx < w, f"Centroid x={cx} out of bounds"
            assert 0 <= cy < h, f"Centroid y={cy} out of bounds"

    def test_source_identifies_detector(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            assert r.metadata["source"] == "OpenCV_RoofPlaneDetector"

    def test_confidence_in_valid_range(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0


class TestRoofPlaneDetectorMultiplePlanes:
    """Detection of multiple distinct regions."""

    def test_detect_two_rectangles(self):
        img = _make_blank()
        _draw_filled_rect(img, 30, 30, 130, 130)
        _draw_filled_rect(img, 250, 250, 370, 370)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        assert len(results) >= 2, f"Expected >=2 planes, got {len(results)}"

    def test_detect_mixed_shapes(self):
        img = _make_blank(500, 500)
        _draw_filled_rect(img, 30, 30, 180, 180)
        _draw_filled_triangle(img, [[300, 300], [400, 300], [350, 400]])
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        assert len(results) >= 2, f"Expected >=2 planes, got {len(results)}"


class TestRoofPlaneDetectorSorting:
    """Results should be sorted by area descending."""

    def test_results_sorted_by_area_descending(self):
        img = _make_blank(500, 500)
        # small rectangle
        _draw_filled_rect(img, 10, 10, 60, 60)
        # big rectangle
        _draw_filled_rect(img, 200, 200, 450, 450)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        if len(results) >= 2:
            areas = [r.metadata["area_pixels"] for r in results]
            for i in range(len(areas) - 1):
                assert areas[i] >= areas[i + 1], "Results not sorted by area descending"


class TestRoofPlaneDetectorParameters:
    """Verify parameter overrides affect behaviour."""

    def test_custom_confidence(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img, confidence_score=0.42)
        for r in results:
            assert r.confidence == pytest.approx(0.42, abs=1e-6)

    def test_high_min_area_filters_small_contours(self):
        img = _make_blank()
        _draw_filled_rect(img, 50, 50, 80, 80)  # small
        detector = RoofPlaneDetector()
        results = detector.detect(img, min_contour_area_ratio=0.5)
        # The small rect is ~900 px in a 160000 px image — well below 50 %
        assert len(results) == 0


class TestRoofPlaneDetectorEdgeCases:
    """Edge-case inputs."""

    def test_uniform_black_image_returns_empty(self):
        img = _make_blank()
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        assert results == []

    def test_uniform_white_image(self):
        img = np.full((400, 400, 3), 255, dtype=np.uint8)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        # A fully white image may produce one large contour or none depending
        # on thresholding; either way should not crash.
        assert isinstance(results, list)

    def test_grayscale_input_raises_or_handles(self):
        """A single-channel image should cause a cv2 error or be handled."""
        gray = np.zeros((400, 400), dtype=np.uint8)
        detector = RoofPlaneDetector()
        # cvtColor(BGR2GRAY) will fail on a 2D array — expect an error
        with pytest.raises(Exception):
            detector.detect(gray)

    def test_very_small_image(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.rectangle(img, (2, 2), (8, 8), (255, 255, 255), -1)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        assert isinstance(results, list)


class TestRoofPlaneDetectorBoundingBox:
    """Ensure bounding box data is coherent."""

    def test_bounding_box_within_image(self):
        h, w = 400, 400
        img = _make_blank(h, w)
        _draw_filled_rect(img, 50, 50, 200, 200)
        detector = RoofPlaneDetector()
        results = detector.detect(img)
        for r in results:
            bb = r.bounding_box
            assert 0 <= bb.x_min < w
            assert 0 <= bb.y_min < h
            assert bb.x_max <= w
            assert bb.y_max <= h
            assert bb.x_min < bb.x_max
            assert bb.y_min < bb.y_max
