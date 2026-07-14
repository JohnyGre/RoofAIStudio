import numpy as np
import cv2

from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector


def test_detect_multiple_polygons():
    # create simple synthetic image with two filled rectangles
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    pts1 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32)
    pts2 = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.int32)
    cv2.fillPoly(img, [pts1], (255, 255, 255))
    cv2.fillPoly(img, [pts2], (255, 255, 255))

    detector = RoofPlaneDetector()
    results = detector.detect(img)

    # Expect at least two detected polygons
    assert results, "No detection results returned"
    assert any(r.metadata and "polygon_vertices" in r.metadata for r in results), "No polygon_vertices in any result metadata"
    # ensure at least two planes found for this synthetic image
    assert len(results) >= 2, f"Expected >=2 planes, got {len(results)}"
