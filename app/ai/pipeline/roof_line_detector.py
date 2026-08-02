"""
Roof line detector component — detects edges, ridges, valleys, and eaves.
Uses HoughLinesP with adaptive thresholding and line grouping.
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass

from app.ai.ai_result import DetectionResult, BoundingBox, LineGeometry
from app.ai.pipeline.base_detector import BaseSubDetector
from app.core.logger import setup_logging
from app.geometry.point import Point2D

logger = setup_logging()


@dataclass
class LineSegment:
    """Raw line segment from Hough detection."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle_deg: float = 0.0
    length: float = 0.0

    def __post_init__(self):
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.length = np.sqrt(dx * dx + dy * dy)
        self.angle_deg = np.degrees(np.arctan2(dy, dx)) % 180


def _group_lines(
    lines: List[LineSegment],
    angle_tolerance_deg: float = 10.0,
    distance_threshold: float = 20.0,
) -> List[List[LineSegment]]:
    """Group collinear line segments that are close together."""
    if not lines:
        return []

    groups: List[List[LineSegment]] = []
    used = [False] * len(lines)

    for i, line in enumerate(lines):
        if used[i]:
            continue
        group = [line]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            other = lines[j]
            # Check angle similarity
            angle_diff = abs(line.angle_deg - other.angle_deg)
            angle_diff = min(angle_diff, 180 - angle_diff)

            if angle_diff > angle_tolerance_deg:
                continue

            # Check distance: minimum distance between endpoints
            def point_line_dist(x, y, seg):
                d1 = np.sqrt((x - seg.x1) ** 2 + (y - seg.y1) ** 2)
                d2 = np.sqrt((x - seg.x2) ** 2 + (y - seg.y2) ** 2)
                # Also check distance to line segment
                dx = seg.x2 - seg.x1
                dy = seg.y2 - seg.y1
                l2 = dx * dx + dy * dy
                if l2 == 0:
                    return min(d1, d2)
                t = max(0, min(1, ((x - seg.x1) * dx + (y - seg.y1) * dy) / l2))
                px = seg.x1 + t * dx
                py = seg.y1 + t * dy
                return np.sqrt((x - px) ** 2 + (y - py) ** 2)

            dists = [
                point_line_dist(other.x1, other.y1, line),
                point_line_dist(other.x2, other.y2, line),
                point_line_dist(line.x1, line.y1, other),
                point_line_dist(line.x2, line.y2, other),
            ]
            min_dist = min(dists)

            if min_dist < distance_threshold:
                group.append(other)
                used[j] = True

        if len(group) >= 1:
            groups.append(group)

    return groups


def _merge_lines(group: List[LineSegment]) -> LineSegment:
    """Merge a group of collinear lines into a single line segment."""
    if len(group) == 1:
        return group[0]

    # Fit a line through all endpoints using PCA
    all_pts = []
    for seg in group:
        all_pts.append([seg.x1, seg.y1])
        all_pts.append([seg.x2, seg.y2])

    pts = np.array(all_pts, dtype=np.float64)

    # Principal component
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Project all points onto principal direction
    projections = centered @ principal
    proj_min = np.min(projections)
    proj_max = np.max(projections)

    start = mean + proj_min * principal
    end = mean + proj_max * principal

    return LineSegment(
        x1=float(np.real(start[0])), y1=float(np.real(start[1])),
        x2=float(np.real(end[0])), y2=float(np.real(end[1])),
    )


class RoofLineDetector(BaseSubDetector):
    """Detects roof lines (ridges, valleys, eaves, edges) using Hough transform."""

    def detect(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> List[DetectionResult]:
        """
        Detect lines in the roof image.

        Args:
            image: BGR numpy array
            **kwargs:
                - hough_rho (default 1)
                - hough_theta (default pi/180)
                - hough_threshold (default 80)
                - hough_min_line_length (default 60)
                - hough_max_line_gap (default 15)
                - angle_group_tolerance_deg (default 10)
                - distance_merge_threshold (default 30)
                - min_line_length_ratio (default 0.1, relative to image diag)
                - confidence_score (default 0.7)

        Returns:
            List[DetectionResult]
        """
        height, width = image.shape[:2]
        image_diag = np.sqrt(height ** 2 + width ** 2)
        min_line_len = int(
            kwargs.get("min_line_length_ratio", 0.1) * image_diag
        )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive Canny with auto-threshold
        edges = cv2.Canny(
            blurred,
            kwargs.get("canny_low", 50),
            kwargs.get("canny_high", 150),
        )

        # Probabilistic Hough
        lines = cv2.HoughLinesP(
            edges,
            rho=kwargs.get("hough_rho", 1),
            theta=kwargs.get("hough_theta", np.pi / 180),
            threshold=kwargs.get("hough_threshold", 80),
            minLineLength=max(min_line_len, kwargs.get("hough_min_line_length", 60)),
            maxLineGap=kwargs.get("hough_max_line_gap", 15),
        )

        if lines is None:
            logger.debug("RoofLineDetector: No Hough lines found.")
            return []

        # Convert to LineSegment objects (handle both (N,4) and (N,1,4) shapes)
        if lines.ndim == 3:
            line_data = lines[:, 0, :]
        else:
            line_data = lines

        segments = [
            LineSegment(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
            )
            for x1, y1, x2, y2 in line_data
        ]

        # Group and merge collinear lines
        groups = _group_lines(
            segments,
            angle_tolerance_deg=kwargs.get("angle_group_tolerance_deg", 10.0),
            distance_threshold=kwargs.get("distance_merge_threshold", 30.0),
        )

        merged = [_merge_lines(g) for g in groups]

        # Filter by length
        merged = [m for m in merged if m.length >= min_line_len * 0.5]

        # Classify: horizontal, vertical, or diagonal
        results: List[DetectionResult] = []
        confidence = kwargs.get("confidence_score", 0.7)

        for seg in merged:
            # Determine line type
            angle = seg.angle_deg
            if angle < 5 or angle > 175:
                line_type = "horizontal_edge"
            elif 85 < angle < 95:
                line_type = "vertical_edge"
            elif angle < 45 or angle > 135:
                line_type = "eave_or_ridge"
            else:
                line_type = "valley_or_hip"

            results.append(DetectionResult(
                geometry=LineGeometry(
                    start_point=(seg.x1, seg.y1),
                    end_point=(seg.x2, seg.y2),
                ),
                confidence=confidence,
                class_name="roof_line",
                metadata={
                    "angle_deg": round(seg.angle_deg, 1),
                    "length_pixels": round(seg.length, 1),
                    "line_type": line_type,
                    "source": "hough_line",
                },
            ))

        logger.info(f"RoofLineDetector: Found {len(results)} lines "
                     f"(from {len(segments)} Hough segments, {len(merged)} merged)")
        return results
