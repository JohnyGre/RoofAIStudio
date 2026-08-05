"""
Polygon Regularizer -- turns jagged SAM masks into clean roof-like shapes.
"""
import cv2
import numpy as np
import logging
from collections import Counter
from typing import List, Tuple

logger = logging.getLogger(__name__)


def regularize_contour(
    contour: np.ndarray,
    epsilon_factor: float = 0.015,
    min_vertices: int = 3,
    max_vertices: int = 12,
    snap_angle_threshold: float = 10.0,
) -> np.ndarray:
    """Regularize a jagged polygon into a clean roof-plane shape."""
    if contour is None or len(contour) < 3:
        return contour

    if contour.ndim == 3:
        points = contour.reshape(-1, 2).astype(np.float32)
    else:
        points = contour.astype(np.float32)

    if len(points) < 3:
        return contour

    # Step 1: Convex hull
    hull = cv2.convexHull(points.astype(np.float32))
    if hull is None or len(hull) < 3:
        return contour
    hull_points = hull.reshape(-1, 2)

    # Step 2: Douglas-Peucker simplification
    perimeter = cv2.arcLength(hull, True)
    epsilon = epsilon_factor * perimeter
    approx = cv2.approxPolyDP(hull, epsilon, True)
    approx_points = approx.reshape(-1, 2)

    # Step 3: Merge short edges
    merged = _merge_short_edges(approx_points, min_edge_fraction=0.12, perimeter=perimeter)

    # Step 4: Snap orthogonal
    snapped = _snap_orthogonal(merged, threshold_deg=snap_angle_threshold)

    # Step 5: Enforce vertex limits
    if len(snapped) > max_vertices:
        new_epsilon = epsilon * 2.0
        approx2 = cv2.approxPolyDP(hull, new_epsilon, True)
        snapped = approx2.reshape(-1, 2)
        if len(snapped) > max_vertices:
            center = np.mean(snapped, axis=0)
            dists = np.linalg.norm(snapped - center, axis=1)
            idx = np.argsort(dists)[-max_vertices:]
            snapped = snapped[np.sort(idx)]

    if len(snapped) < min_vertices:
        snapped = hull_points

    result = snapped.astype(np.int32)
    if contour.ndim == 3:
        result = result.reshape(-1, 1, 2)
    return result


def _merge_short_edges(points, min_edge_fraction=0.12, perimeter=0):
    if len(points) < 4:
        return points
    n = len(points)
    if perimeter <= 0:
        perimeter = sum(np.linalg.norm(points[i] - points[(i+1)%n]) for i in range(n))
    min_edge = min_edge_fraction * perimeter
    keep = [True] * n
    for i in range(n):
        edge_len = np.linalg.norm(points[i] - points[(i+1)%n])
        if edge_len < min_edge and edge_len > 0:
            if keep[i]:
                keep[(i+1)%n] = False
    result = points[keep]
    return result if len(result) >= 3 else points


def _snap_orthogonal(points, threshold_deg=10.0):
    if len(points) < 3:
        return points
    n = len(points)
    snapped = points.copy()
    threshold_rad = np.deg2rad(threshold_deg)

    def _snap_to_90(ang):
        return round(ang / (np.pi/2)) * (np.pi/2)

    for i in range(n):
        j = (i+1) % n
        dx = points[j,0] - points[i,0]
        dy = points[j,1] - points[i,1]
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            continue
        edge_angle = np.arctan2(dy, dx)
        nearest_ortho = _snap_to_90(edge_angle)
        diff = abs(edge_angle - nearest_ortho)
        if diff > np.pi:
            diff = 2*np.pi - diff
        if diff < threshold_rad:
            edge_len = np.linalg.norm([dx, dy])
            new_dx = edge_len * np.cos(nearest_ortho)
            new_dy = edge_len * np.sin(nearest_ortho)
            if i > 0:
                snapped[j] = snapped[i] + np.array([new_dx, new_dy])
    return snapped