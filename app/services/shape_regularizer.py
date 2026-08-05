"""
Shape Regularizer — fits contour polygons to architectural roof shapes.

Roof planes have well-defined geometries:
- Triangles (valby / nárožné trojuholníky): 2 sides often equal (isosceles), ridge vertex at top
- Rectangles: 4 right angles, parallel opposite edges
- Trapezoids: 4 vertices, 2 parallel edges (top/bottom or left/right)
- Gable pairs (sedlové strechy): mirrored shapes sharing a ridge edge

Pipeline per contour:
1. Douglas-Peucker → vertex count
2. Classify shape (triangle / quadrilateral / complex)
3. Fit best geometric primitive
4. Post-pairing: snap shared edges for gable roofs
"""
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def fit_triangle(pts: np.ndarray) -> np.ndarray:
    """
    Fit an isosceles triangle to 3 points.
    Looks for a pair of nearly-equal sides (the hip edges) and enforces equality.

    For roof triangles (valby), the two longer sides meeting at the ridge
    should be equal. The base (eave) may differ.
    """
    if len(pts) != 3:
        return pts

    d01 = np.linalg.norm(pts[1] - pts[0])
    d12 = np.linalg.norm(pts[2] - pts[1])
    d20 = np.linalg.norm(pts[0] - pts[2])
    sides = [(d01, 0, 1), (d12, 1, 2), (d20, 2, 0)]

    # Find the two sides closest in length → these should be equal (hip edges)
    sorted_sides = sorted(sides, key=lambda s: s[0])
    # Two smaller and one larger → the two smaller are often the hip edges
    # Find the most similar pair
    best_pair = None
    best_diff = float("inf")
    pairs = [(0, 1), (1, 2), (0, 2)]
    for a, b in pairs:
        diff = abs(sides[a][0] - sides[b][0])
        if diff < best_diff:
            best_diff = diff
            best_pair = (a, b)

    if best_pair and best_diff < sides[best_pair[0]][0] * 0.5:
        # Enforce equal length for the paired sides
        avg_len = (sides[best_pair[0]][0] + sides[best_pair[1]][0]) / 2.0

        # The third side is the unique one (base)
        all_idx = {0, 1, 2}
        paired_idxs = {best_pair[0], best_pair[1]}
        base_idx = list(all_idx - paired_idxs)[0]

        # Reconstruct triangle with two equal sides
        base_s = sides[base_idx]
        base_len = base_s[0]
        base_start = base_s[1]
        base_end = base_s[2]

        # Apex is the vertex not on the base
        apex_idx = list(set(range(3)) - {base_start, base_end})[0]
        base_mid = (pts[base_start] + pts[base_end]) / 2.0
        base_dir = pts[base_end] - pts[base_start]
        base_normal = np.array([-base_dir[1], base_dir[0]])
        base_normal = base_normal / (np.linalg.norm(base_normal) + 1e-8)

        # Height from apex to base
        h = np.sqrt(max(avg_len ** 2 - (base_len / 2) ** 2, 0))
        # Direction: keep apex on same side of base
        sign = np.sign(np.dot(pts[apex_idx] - base_mid, base_normal))
        new_apex = base_mid + base_normal * h * sign

        result = np.array([pts[base_start], pts[base_end], new_apex])
        return result.astype(np.int32)

    return pts.astype(np.int32)


def fit_rectangle(pts: np.ndarray) -> np.ndarray:
    """
    Fit a rectangle (4 right angles, parallel opposite edges) to 4 points.
    Uses the minimum area bounding rotated rectangle.
    """
    if len(pts) < 4:
        return pts

    # Use the 4 most salient vertices
    if len(pts) > 4:
        hull = cv2.convexHull(pts)
        if hull is not None and len(hull) >= 4:
            hull2d = hull.reshape(-1, 2).astype(np.float32)
            peri = cv2.arcLength(hull, True)
            pts = cv2.approxPolyDP(hull2d, 0.02 * peri, True).astype(np.float32)
        else:
            pts = pts[:4]

    if len(pts) != 4:
        return pts

    # Fit rotated bounding rectangle
    rect = cv2.minAreaRect(pts.astype(np.float32))
    box = cv2.boxPoints(rect)
    return box.astype(np.int32).reshape(-1, 1, 2)


def fit_trapezoid(pts: np.ndarray) -> np.ndarray:
    """
    Fit a trapezoid: 4 vertices with exactly two parallel edges.

    The parallel edges are usually the ridge (top) and eave (bottom),
    or two side edges for hip roofs.
    """
    if len(pts) < 4:
        return pts

    if len(pts) > 4:
        hull = cv2.convexHull(pts)
        if hull is not None and len(hull) >= 4:
            hull2d = hull.reshape(-1, 2).astype(np.float32)
            peri = cv2.arcLength(hull, True)
            pts = cv2.approxPolyDP(hull2d, 0.02 * peri, True).astype(np.float32)

    pts = pts[:4]
    if len(pts) != 4:
        return pts

    # Find the pair of most parallel edges
    n = 4
    best_parallel_pair = None
    best_parallel_score = float("inf")

    # Edge pairs: (0-1,2-3) or (1-2,3-0)
    edge_pairs = [((0, 1), (2, 3)), ((1, 2), (3, 0))]

    for (sa, ea), (sb, eb) in edge_pairs:
        v1 = pts[ea] - pts[sa]
        v2 = pts[eb] - pts[sb]
        # Flatten in case pts is (N,1,2)
        v1 = v1.flatten()
        v2 = v2.flatten()
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1 or len2 < 1:
            continue

        # Dot product: 1.0 = parallel, 0.0 = perpendicular
        dot = abs(np.dot(v1 / len1, v2 / len2))
        # We want parallel → dot close to 1
        score = 1.0 - dot
        if score < best_parallel_score:
            best_parallel_score = score
            best_parallel_pair = ((sa, ea), (sb, eb))

    if best_parallel_pair is None:
        return pts.reshape(-1, 1, 2).astype(np.int32)

    # Enforce parallelism on the best pair
    (sa, ea), (sb, eb) = best_parallel_pair
    v1 = pts[ea] - pts[sa]
    v2 = pts[eb] - pts[sb]

    # Make both edges parallel to the average direction
    avg_dir = (v1 / max(np.linalg.norm(v1), 1) + v2 / max(np.linalg.norm(v2), 1))
    avg_dir = avg_dir / max(np.linalg.norm(avg_dir), 1)

    # Project the edge lengths onto the average direction
    new_v1 = avg_dir * np.linalg.norm(v1)
    new_v2 = avg_dir * np.linalg.norm(v2)

    result = pts.copy()
    result[ea] = result[sa] + new_v1
    result[eb] = result[sb] + new_v2

    # Re-order: ensure edges are connected
    return result.astype(np.int32).reshape(-1, 1, 2)


def regularize_shape(
    contour: np.ndarray,
    epsilon_factor: float = 0.02,
) -> np.ndarray:
    """
    Full shape-fitting pipeline for a single contour.

    Returns a regularized contour in (N, 1, 2) int32 format.
    """
    if contour is None or len(contour) < 3:
        return contour

    if contour.ndim == 3:
        pts = contour.reshape(-1, 2).astype(np.float32)
    else:
        pts = contour.astype(np.float32)

    if len(pts) < 3:
        return contour

    # Step 1: Convex hull (roof planes are convex)
    hull_raw = cv2.convexHull(pts)
    if hull_raw is None or len(hull_raw) < 3:
        return contour
    hull = hull_raw.reshape(-1, 2).astype(np.float32)

    # Step 2: Douglas-Peucker to get key vertices
    peri = cv2.arcLength(hull_raw, True)
    approx = cv2.approxPolyDP(hull_raw, epsilon_factor * peri, True)
    verts = approx.reshape(-1, 2).astype(np.float32)
    n = len(verts)

    # Step 3: Shape classification + fitting
    if n == 3:
        result = fit_triangle(verts)
    elif n == 4:
        # Check if it's closer to rectangle or trapezoid
        angles = []
        for i in range(4):
            j = (i + 1) % 4
            k = (i + 2) % 4
            v1 = verts[j] - verts[i]
            v2 = verts[k] - verts[j]
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)
            if l1 > 0 and l2 > 0:
                cos_a = np.dot(v1, v2) / (l1 * l2)
                angles.append(abs(cos_a))

        # If all angles are ~90° (cos ≈ 0), it's a rectangle
        is_rect = all(a < 0.25 for a in angles)  # cos(75°) ≈ 0.26

        if is_rect:
            result = fit_rectangle(verts)
        else:
            result = fit_trapezoid(verts)
    elif n <= 6:
        # Polygon: fit as trapezoid or simplified polygon
        result = fit_trapezoid(verts)
    else:
        result = verts.astype(np.int32)

    # Ensure result is (N, 1, 2)
    if result.ndim == 2:
        result = result.reshape(-1, 1, 2)

    return result.astype(np.int32)


def snap_shared_edges(planes: List[dict], snap_threshold_px: float = 15.0):
    """
    Find gable roof pairs and snap their shared ridge edge.

    Two planes form a gable pair if:
    1. They have a nearly-coincident edge
    2. They are on opposite sides of that shared edge (mirrored)

    Snaps the shared edge to the average line.
    """
    if len(planes) < 2:
        return

    contours = [p.get("regularized_contour") for p in planes]
    if any(c is None or len(c) < 3 for c in contours):
        return

    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            ci = contours[i]
            cj = contours[j]
            if ci is None or cj is None:
                continue

            pi = ci.reshape(-1, 2).astype(np.float32)
            pj = cj.reshape(-1, 2).astype(np.float32)

            # Find closest pair of edges between the two contours
            best_edge_dist = float("inf")
            best_pair = None
            for ei in range(len(pi)):
                ei_next = (ei + 1) % len(pi)
                ei_pt1 = pi[ei]
                ei_pt2 = pi[ei_next]
                ei_dir = ei_pt2 - ei_pt1
                ei_len = np.linalg.norm(ei_dir)

                for ej in range(len(pj)):
                    ej_next = (ej + 1) % len(pj)
                    ej_pt1 = pj[ej]
                    ej_pt2 = pj[ej_next]
                    ej_dir = ej_pt2 - ej_pt1
                    ej_len = np.linalg.norm(ej_dir)

                    # Check: edges should be similar length and close
                    if abs(ei_len - ej_len) > ei_len * 0.3:
                        continue

                    # Distance between edge midpoints
                    ei_mid = (ei_pt1 + ei_pt2) / 2
                    ej_mid = (ej_pt1 + ej_pt2) / 2
                    mid_dist = np.linalg.norm(ei_mid - ej_mid)

                    if mid_dist < snap_threshold_px and mid_dist < best_edge_dist:
                        best_edge_dist = mid_dist
                        best_pair = (i, ei, ei_next, j, ej, ej_next, ei_mid, ej_mid)

            if best_pair and best_edge_dist < snap_threshold_px:
                # Snap: move both edges to their average position
                i_idx, ei, ei_n, j_idx, ej, ej_n, mid_i, mid_j = best_pair
                avg_mid = (mid_i + mid_j) / 2

                # Compute displacement vectors
                disp_i = avg_mid - mid_i
                disp_j = avg_mid - mid_j

                pi_new = pi.copy()
                pj_new = pj.copy()
                pi_new[ei] += disp_i
                pi_new[ei_n] += disp_i
                pj_new[ej] += disp_j
                pj_new[ej_n] += disp_j

                # Store back
                planes[i_idx]["regularized_contour"] = pi_new.astype(np.int32).reshape(-1, 1, 2)
                planes[j_idx]["regularized_contour"] = pj_new.astype(np.int32).reshape(-1, 1, 2)
                planes[i_idx]["paired_with"] = j_idx
                planes[j_idx]["paired_with"] = i_idx
                planes[i_idx]["shared_edge_length_px"] = np.linalg.norm(pi_new[ei] - pi_new[ei_n])
                planes[j_idx]["shared_edge_length_px"] = planes[i_idx]["shared_edge_length_px"]
