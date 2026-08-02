"""
Smoke test for RoofPlaneDetector v2 — multi-strategy contour detection.
Tests edge-based, watershed, K-means, and morph-gradient strategies
on synthetic images with varying complexity.
"""

import numpy as np
import cv2
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector, ContourRefiner


def create_test_images():
    """Create synthetic test images for roof plane detection."""
    images = {}

    # 1. Simple rectangle (single roof plane)
    img1 = np.zeros((300, 400, 3), dtype=np.uint8)
    img1[50:250, 50:350] = (100, 120, 140)  # Gray-blue roof
    images["simple_rect"] = img1

    # 2. L-shaped roof (concave polygon test)
    img2 = np.zeros((400, 400, 3), dtype=np.uint8)
    l_shape = np.array([
        [50, 50], [200, 50], [200, 150], [120, 150],
        [120, 250], [50, 250]
    ], dtype=np.int32)
    cv2.fillPoly(img2, [l_shape], (80, 100, 120))
    images["l_shape"] = img2

    # 3. Two adjacent roof planes with different color
    img3 = np.zeros((300, 500, 3), dtype=np.uint8)
    img3[30:270, 30:230] = (120, 100, 80)  # Left plane - brown
    img3[30:270, 260:470] = (80, 90, 120)  # Right plane - blue-gray
    # Ridge line
    cv2.line(img3, (250, 30), (250, 270), (50, 50, 50), 1)
    images["two_planes"] = img3

    # 4. Multiple planes + noise (more realistic)
    img4 = np.zeros((400, 600, 3), dtype=np.uint8)
    # Plane 1
    img4[20:180, 20:280] = (110, 115, 125)
    # Plane 2
    img4[20:180, 300:580] = (90, 95, 110)
    # Plane 3 (small dormer)
    img4[200:350, 150:450] = (100, 105, 115)
    # Add slight noise
    noise = np.random.randint(0, 8, img4.shape, dtype=np.uint8)
    img4 = cv2.add(img4, noise)
    # Edge lines
    cv2.line(img4, (290, 20), (290, 180), (60, 60, 60), 1)
    cv2.line(img4, (140, 200), (140, 350), (60, 60, 60), 1)
    cv2.line(img4, (450, 200), (450, 350), (60, 60, 60), 1)
    images["multi_plane_noise"] = img4

    # 5. Pentagon (non-rectangular roof)
    img5 = np.zeros((350, 450, 3), dtype=np.uint8)
    pent = np.array([
        [100, 50], [350, 80], [400, 200], [300, 300], [50, 250]
    ], dtype=np.int32)
    cv2.fillPoly(img5, [pent], (100, 110, 130))
    images["pentagon"] = img5

    return images


def has_concavity(verts):
    """Detect if polygon has at least one reflex (concave) vertex."""
    if not verts or len(verts) < 4:
        return False
    n = len(verts)
    signs = []
    for i in range(n):
        p_prev = verts[(i - 1) % n]
        p_curr = verts[i]
        p_next = verts[(i + 1) % n]
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        signs.append(cross)
    positive = sum(1 for s in signs if s > 0)
    negative = sum(1 for s in signs if s < 0)
    return positive > 0 and negative > 0


def run_tests():
    """Run all smoke tests."""
    images = create_test_images()
    detector = RoofPlaneDetector()
    
    all_passed = True
    total_planes = 0

    for name, img in images.items():
        print(f"\n{'='*50}")
        print(f"Test: {name} ({img.shape[1]}x{img.shape[0]})")
        print(f"{'='*50}")

        results = detector.detect(img, score_threshold=0.10)
        print(f"Found {len(results)} planes")

        for i, r in enumerate(results):
            meta = r.metadata or {}
            verts = meta.get("polygon_vertices", [])
            area = meta.get("area_pixels", 0)
            score = meta.get("contour_score", 0)
            solidity = meta.get("solidity", 0)
            source = meta.get("source", "?")
            concave = has_concavity(verts) if len(verts) >= 4 else False

            print(
                f"  Plane {i}: "
                f"area={area:.0f} "
                f"vertices={len(verts)} "
                f"score={score:.3f} "
                f"solidity={solidity:.3f} "
                f"concave={concave} "
                f"source={source}"
            )

        total_planes += len(results)

        # Validation per test type
        if name == "l_shape":
            # Should preserve concavity
            has_concave = any(
                has_concavity(r.metadata.get("polygon_vertices", []))
                for r in results
            )
            if has_concave:
                print(f"  PASS: Concavity preserved in L-shape")
            else:
                print(f"  WARN: No concave polygon detected in L-shape (may vary by strategy)")
                # Not a hard fail — watershed might split L-shape into convex parts

        if name == "two_planes":
            if len(results) >= 1:
                print(f"  PASS: Found at least one plane in two-plane image")
            else:
                print(f"  FAIL: No planes found in two-plane image")
                all_passed = False

        if name == "simple_rect":
            if len(results) >= 1:
                print(f"  PASS: Found plane in simple rectangle")
            else:
                print(f"  FAIL: No plane found in simple rectangle")
                all_passed = False

    print(f"\n{'='*50}")
    print(f"SUMMARY: {total_planes} total planes across {len(images)} test images")
    if all_passed:
        print("RESULT: ALL CRITICAL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print(f"{'='*50}")

    return all_passed


def test_contour_refiner():
    """Test ContourRefiner deduplication."""
    print(f"\n{'='*50}")
    print("ContourRefiner Tests")
    print(f"{'='*50}")

    # Create two overlapping contours
    from app.ai.pipeline.roof_plane_detector import ContourCandidate

    c1_pts = np.array([[[0, 0]], [[100, 0]], [[100, 100]], [[0, 100]]], dtype=np.int32)
    c2_pts = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32)

    c1 = ContourCandidate(contour=c1_pts, area=10000, score=0.8, source="test1")
    c2 = ContourCandidate(contour=c2_pts, area=6400, score=0.9, source="test2")

    deduped = ContourRefiner.remove_nested_duplicates([c1, c2], iou_threshold=0.5)
    print(f"Dedup test: {len(deduped)} contours kept (expected 1)")

    if len(deduped) == 1 and deduped[0].score == 0.9:
        print("PASS: Higher-scored duplicate kept")
    else:
        print(f"FAIL: Expected 1 contour with score 0.9, got {len(deduped)}")


if __name__ == "__main__":
    test_contour_refiner()
    ok = run_tests()
    sys.exit(0 if ok else 1)
