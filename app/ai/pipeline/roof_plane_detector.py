"""
Roof plane detector — improved multi-strategy contour detection.
Combines edge-based, region-based, and morphological approaches
to reliably detect roof plane outlines without requiring dimensions.
"""

import uuid
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from app.ai.ai_result import DetectionResult, BoundingBox, BBoxGeometry, LineGeometry
from app.core.logger import setup_logging
from app.geometry.point import Point2D

logger = setup_logging()


@dataclass
class ContourCandidate:
    """Intermediate contour with quality scores for filtering."""
    contour: np.ndarray
    area: float = 0.0
    perimeter: float = 0.0
    solidity: float = 0.0
    rectangularity: float = 0.0
    edge_alignment: float = 0.0
    mean_boundary_contrast: float = 0.0
    score: float = 0.0
    source: str = "unknown"


class ContourRefiner:
    """Post-processing utilities for contour cleanup and merging."""

    @staticmethod
    def close_gaps(contour: np.ndarray, max_gap_ratio: float = 0.05, image_diag: float = 1.0) -> np.ndarray:
        """Close small gaps in a contour by connecting nearby endpoints."""
        if len(contour) < 4:
            return contour
        
        max_gap = max_gap_ratio * image_diag
        if cv2.arcLength(contour, True) > 0 and cv2.contourArea(contour) > 0:
            return contour
        
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            return contour

        start_pt = pts[0]
        end_pt = pts[-1]
        dist = np.linalg.norm(end_pt - start_pt)
        if dist < max_gap and len(pts) >= 3:
            closed = np.vstack([pts, pts[0:1]])
            return closed.reshape(-1, 1, 2).astype(np.int32)

        return contour

    @staticmethod
    def remove_nested_duplicates(candidates: List[ContourCandidate], iou_threshold: float = 0.85) -> List[ContourCandidate]:
        """Remove near-duplicate contours (high IoU overlap). Keeps highest-scoring."""
        if len(candidates) <= 1:
            return candidates

        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        keep = []
        kept_bboxes = []

        for c in candidates:
            bx, by, bw, bh = cv2.boundingRect(c.contour)
            is_duplicate = False
            for (kbx, kby, kbw, kbh) in kept_bboxes:
                ix = max(bx, kbx)
                iy = max(by, kby)
                iw = min(bx + bw, kbx + kbw) - ix
                ih = min(by + bh, kby + kbh) - iy
                if iw <= 0 or ih <= 0:
                    continue
                intersection = iw * ih
                union = (bw * bh) + (kbw * kbh) - intersection
                iou = intersection / union if union > 0 else 0
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(c)
                kept_bboxes.append((bx, by, bw, bh))

        return keep

    @staticmethod
    def snap_to_right_angles(contour: np.ndarray, angle_tolerance_deg: float = 10.0) -> np.ndarray:
        """Snap approximately orthogonal corners to exact right angles."""
        if len(contour) < 4:
            return contour

        pts = contour.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        adjusted = pts.copy()

        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]

            v1 = a - b
            v2 = c - b
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))

            if abs(angle_deg - 90.0) < angle_tolerance_deg:
                d = a - b
                d_perp = np.array([-d[1], d[0]])
                d_perp_norm = np.linalg.norm(d_perp)
                if d_perp_norm > 1e-6:
                    d_perp = d_perp / d_perp_norm
                    proj = np.dot(c - b, d_perp)
                    adjusted[i] = b + d_perp * (abs(proj) if proj > 0 else abs(proj))
                    adjusted[i] = b + d_perp * abs(proj)

        return adjusted.astype(np.int32).reshape(-1, 1, 2)


class RoofPlaneDetector:
    """
    Detects individual roof plane outlines using a multi-strategy approach:
    1. Edge-based detection (improved adaptive Canny)
    2. Region-based segmentation (watershed)
    3. Color clustering (K-means)
    4. Morphological gradient detection
    
    All strategies run in parallel and results are merged with deduplication.
    """

    def __init__(self):
        logger.info("RoofPlaneDetector v2 initialized (multi-strategy).")

    # ── Strategy 1: Edge-based ────────────────────────────────────────────

    def _detect_edge_based(
        self, image: np.ndarray, height: int, width: int, image_area: float, **kwargs
    ) -> List[ContourCandidate]:
        """Improved edge-based detection with multi-scale Canny."""
        candidates: List[ContourCandidate] = []

        blur_kernel = kwargs.get("blur_kernel_size", 5)
        canny_low_base = kwargs.get("canny_threshold1", 20)
        canny_high_base = kwargs.get("canny_threshold2", 80)
        morph_kernel = kwargs.get("morph_kernel_size", 3)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Multi-scale Canny
        scale_factors = [0.6, 1.0, 1.5]
        combined_edges = np.zeros_like(blurred)

        for sf in scale_factors:
            low = int(canny_low_base * sf)
            high = int(canny_high_base * sf)
            edges = cv2.Canny(blurred, low, high)
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            combined_edges = cv2.bitwise_or(combined_edges, edges)

        # Close gaps
        kernel = np.ones((morph_kernel + 2, morph_kernel + 2), np.uint8)
        closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # RETR_TREE for nested planes
        contours, hierarchy = cv2.findContours(
            closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if hierarchy is None:
            return candidates

        hierarchy = hierarchy[0]
        for i, h in enumerate(hierarchy):
            if i >= len(contours):
                break
            area = cv2.contourArea(contours[i])
            min_area = 0.0005 * image_area
            max_area = 0.98 * image_area
            if not (min_area <= area <= max_area):
                continue

            perimeter = cv2.arcLength(contours[i], True)
            if perimeter < 1:
                continue

            approx = cv2.approxPolyDP(contours[i], 0.02 * perimeter, True)
            if len(approx) < 3:
                continue

            # Solidity
            hull = cv2.convexHull(contours[i])
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Rectangularity
            bx, by, bw, bh = cv2.boundingRect(contours[i])
            rect_area = bw * bh
            rectangularity = area / rect_area if rect_area > 0 else 0

            # Boundary contrast
            grad = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(grad, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(grad, cv2.CV_64F, 0, 1, ksize=3)
            gmag = cv2.magnitude(gx, gy)
            gmag = np.uint8(cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX))
            bc = RoofPlaneDetector._compute_boundary_contrast(gmag, contours[i])

            # Nesting level
            level = 0
            parent_idx = h[3]
            while parent_idx != -1 and level < 10:
                level += 1
                parent_idx = hierarchy[parent_idx][3]

            candidates.append(ContourCandidate(
                contour=approx,
                area=area,
                perimeter=perimeter,
                solidity=solidity,
                rectangularity=rectangularity,
                mean_boundary_contrast=bc,
                source=f"edge_L{level}",
            ))

        return candidates

    # ── Strategy 2: Watershed ─────────────────────────────────────────────

    def _detect_region_based(
        self, image: np.ndarray, height: int, width: int, image_area: float, **kwargs
    ) -> List[ContourCandidate]:
        """Watershed-based segmentation to find homogeneous roof regions."""
        candidates: List[ContourCandidate] = []

        blur_kernel = kwargs.get("blur_kernel_size", 7)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Gradient magnitude
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = np.uint8(cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX))

        # Threshold to find flat regions
        _, markers_bin = cv2.threshold(
            gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        markers_bin = cv2.bitwise_not(markers_bin)

        kernel = np.ones((3, 3), np.uint8)
        markers_bin = cv2.morphologyEx(markers_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        markers_bin = cv2.morphologyEx(markers_bin, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Distance transform
        dist = cv2.distanceTransform(markers_bin, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

        for marker_thresh in [0.3, 0.2, 0.15]:
            _, sure_fg = cv2.threshold(
                (dist_norm * 255).astype(np.uint8), int(marker_thresh * 255), 255, cv2.THRESH_BINARY
            )
            sure_fg = np.uint8(sure_fg)

            num_markers, markers = cv2.connectedComponents(sure_fg)

            if num_markers <= 1:
                continue

            markers = markers + 1
            unknown = cv2.subtract(markers_bin, sure_fg)
            markers[unknown == 255] = 0

            watershed_img = image.copy()
            markers = cv2.watershed(watershed_img, markers)

            for label in range(2, markers.max() + 1):
                region_mask = np.uint8(markers == label) * 255
                region_contours, _ = cv2.findContours(
                    region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for rc in region_contours:
                    area = cv2.contourArea(rc)
                    min_area = 0.001 * image_area
                    max_area = 0.95 * image_area
                    if not (min_area <= area <= max_area):
                        continue

                    perimeter = cv2.arcLength(rc, True)
                    approx = cv2.approxPolyDP(rc, 0.03 * perimeter, True)
                    if len(approx) < 3:
                        continue

                    hull = cv2.convexHull(rc)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0

                    bx, by, bw, bh = cv2.boundingRect(rc)
                    rect_area = bw * bh
                    rectangularity = area / rect_area if rect_area > 0 else 0

                    bc = RoofPlaneDetector._compute_boundary_contrast(gradient, rc)

                    candidates.append(ContourCandidate(
                        contour=approx,
                        area=area,
                        perimeter=perimeter,
                        solidity=solidity,
                        rectangularity=rectangularity,
                        mean_boundary_contrast=bc,
                        source=f"watershed_t{marker_thresh}",
                    ))

        return candidates

    # ── Strategy 3: K-means color clustering ──────────────────────────────

    def _detect_color_cluster_based(
        self, image: np.ndarray, height: int, width: int, image_area: float, **kwargs
    ) -> List[ContourCandidate]:
        """K-means color clustering to find regions of similar color."""
        candidates: List[ContourCandidate] = []

        blur_kernel = kwargs.get("blur_kernel_size", 7)
        n_clusters = kwargs.get("color_clusters", 4)

        blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        pixels = blurred.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            pixels, n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.reshape(height, width)

        # Gradient for boundary contrast
        grad = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(grad, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(grad, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(gx, gy)
        gradient_mag = np.uint8(cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX))

        for cluster_id in range(n_clusters):
            cluster_mask = np.uint8(labels == cluster_id) * 255

            kernel = np.ones((5, 5), np.uint8)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

            contours, _ = cv2.findContours(
                cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                min_area = 0.0005 * image_area
                max_area = 0.95 * image_area
                if not (min_area <= area <= max_area):
                    continue

                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                if len(approx) < 3:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                bx, by, bw, bh = cv2.boundingRect(contour)
                rect_area = bw * bh
                rectangularity = area / rect_area if rect_area > 0 else 0

                bc = RoofPlaneDetector._compute_boundary_contrast(gradient_mag, contour)

                candidates.append(ContourCandidate(
                    contour=approx,
                    area=area,
                    perimeter=perimeter,
                    solidity=solidity,
                    rectangularity=rectangularity,
                    mean_boundary_contrast=bc,
                    source=f"kmeans_C{cluster_id}",
                ))

        return candidates

    # ── Strategy 4: Morphological gradient ────────────────────────────────

    def _detect_morph_gradient_based(
        self, image: np.ndarray, height: int, width: int, image_area: float, **kwargs
    ) -> List[ContourCandidate]:
        """Morphological gradient edge detection."""
        candidates: List[ContourCandidate] = []

        blur_kernel = kwargs.get("blur_kernel_size", 5)
        morph_kernel = kwargs.get("morph_kernel_size", 5)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        gradient_morph = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

        _, thresh = cv2.threshold(
            gradient_morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel_small = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)
            min_area = 0.0003 * image_area
            max_area = 0.98 * image_area
            if not (min_area <= area <= max_area):
                continue

            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) < 3:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            bx, by, bw, bh = cv2.boundingRect(contour)
            rect_area = bw * bh
            rectangularity = area / rect_area if rect_area > 0 else 0

            candidates.append(ContourCandidate(
                contour=approx,
                area=area,
                perimeter=perimeter,
                solidity=solidity,
                rectangularity=rectangularity,
                source="morph_gradient",
            ))

        return candidates

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_boundary_contrast(gradient_map: np.ndarray, contour: np.ndarray) -> float:
        """Compute mean gradient magnitude along the contour boundary."""
        mask = np.zeros(gradient_map.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, 1)
        boundary_pixels = gradient_map[mask == 255]
        if len(boundary_pixels) == 0:
            return 0.0
        return float(np.mean(boundary_pixels)) / 255.0

    @staticmethod
    def _score_candidate(
        c: ContourCandidate,
        min_vertices: int = 3,
        min_area_ratio: float = 0.001,
        image_area: float = 1.0,
    ) -> float:
        """Compute composite quality score for a contour candidate."""
        if c.area < min_area_ratio * image_area or len(c.contour) < min_vertices:
            return 0.0

        # Area score: bell curve peaking at ~15% of image area
        area_ratio = c.area / image_area
        area_score = np.exp(-((area_ratio - 0.15) ** 2) / (2 * 0.12 ** 2))

        # Combined score
        score = (
            0.15 * area_score
            + 0.30 * c.solidity
            + 0.25 * c.rectangularity
            + 0.30 * c.mean_boundary_contrast
        )
        return max(0.0, min(1.0, score))

    # ── Main detect ───────────────────────────────────────────────────────

    def detect(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """
        Multi-strategy roof plane contour detection.

        Args:
            image: BGR numpy array
            **kwargs:
                - blur_kernel_size (default 5)
                - canny_threshold1 (default 20)
                - canny_threshold2 (default 80)
                - morph_kernel_size (default 3)
                - min_contour_area_ratio (default 0.0005)
                - max_contour_area_ratio (default 0.98)
                - min_vertices (default 3)
                - confidence_score (default 0.9)
                - enable_watershed (default True)
                - enable_kmeans (default True)
                - enable_morph_gradient (default True)
                - score_threshold (default 0.15)
                - dedup_iou_threshold (default 0.85)

        Returns:
            List[DetectionResult]
        """
        height, width = image.shape[:2]
        image_area = float(height * width)
        confidence_score = kwargs.get("confidence_score", 0.9)

        # Auto-detect if this is a real photograph (many colors/textures)
        # vs a synthetic image (few flat colors). Use stricter params for real photos.
        if kwargs.get("strict_mode", None) is None:
            gray_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_density = np.mean(cv2.Canny(gray_check, 50, 150) > 0)
            is_real_photo = edge_density > 0.05
        else:
            is_real_photo = kwargs["strict_mode"]

        if is_real_photo:
            logger.debug("Detected real photo — using strict filtering")
            kwargs.setdefault("score_threshold", 0.35)
            kwargs.setdefault("min_contour_area_ratio", 0.01)
            kwargs.setdefault("min_vertices", 4)
            kwargs.setdefault("max_planes", 12)
            kwargs.setdefault("enable_morph_gradient", False)
            kwargs.setdefault("enable_watershed", False)
            kwargs.setdefault("enable_kmeans", False)
            kwargs.setdefault("dedup_iou_threshold", 0.65)
        else:
            kwargs.setdefault("max_planes", 20)

        # Now read all params AFTER strict mode may have set defaults
        min_vertices = kwargs.get("min_vertices", 3)
        min_area_ratio = kwargs.get("min_contour_area_ratio", 0.0005)
        score_threshold = kwargs.get("score_threshold", 0.15)
        max_planes = kwargs.get("max_planes", 20)

        all_candidates: List[ContourCandidate] = []

        # Strategy 1: Edge-based (always run)
        logger.debug("Running edge-based detection...")
        edge_candidates = self._detect_edge_based(image, height, width, image_area, **kwargs)
        all_candidates.extend(edge_candidates)
        logger.debug(f"Edge-based: {len(edge_candidates)} candidates")

        # Strategy 2: Watershed
        if kwargs.get("enable_watershed", True):
            logger.debug("Running watershed-based detection...")
            ws_candidates = self._detect_region_based(image, height, width, image_area, **kwargs)
            all_candidates.extend(ws_candidates)
            logger.debug(f"Watershed: {len(ws_candidates)} candidates")

        # Strategy 3: K-means color clustering
        if kwargs.get("enable_kmeans", True):
            logger.debug("Running color-cluster-based detection...")
            cc_candidates = self._detect_color_cluster_based(image, height, width, image_area, **kwargs)
            all_candidates.extend(cc_candidates)
            logger.debug(f"K-means: {len(cc_candidates)} candidates")

        # Strategy 4: Morphological gradient
        if kwargs.get("enable_morph_gradient", True):
            logger.debug("Running morphological gradient detection...")
            mg_candidates = self._detect_morph_gradient_based(image, height, width, image_area, **kwargs)
            all_candidates.extend(mg_candidates)
            logger.debug(f"Morph gradient: {len(mg_candidates)} candidates")

        # Score all candidates
        for c in all_candidates:
            c.score = self._score_candidate(
                c,
                min_vertices=min_vertices,
                min_area_ratio=min_area_ratio,
                image_area=image_area,
            )

        # Filter by score threshold
        all_candidates = [c for c in all_candidates if c.score >= score_threshold]

        # Deduplicate
        dedup_iou = kwargs.get("dedup_iou_threshold", 0.85)
        unique_candidates = ContourRefiner.remove_nested_duplicates(
            all_candidates, iou_threshold=dedup_iou
        )

        # Sort by score (highest first)
        unique_candidates.sort(key=lambda c: c.score, reverse=True)

        # Apply max planes cap
        max_planes = kwargs.get("max_planes", 20)
        if len(unique_candidates) > max_planes:
            unique_candidates = unique_candidates[:max_planes]

        # Convert to DetectionResult
        results: List[DetectionResult] = []
        for c in unique_candidates:
            pts_raw = c.contour.reshape(-1, 2)
            valid_pts = []
            for px, py in pts_raw:
                px_f, py_f = float(px), float(py)
                if 0 <= px_f < width and 0 <= py_f < height:
                    valid_pts.append(Point2D(px_f, py_f))

            if len(valid_pts) < min_vertices:
                continue

            bx, by, bw, bh = cv2.boundingRect(c.contour)
            geometry = BBoxGeometry(
                x_min=float(bx), y_min=float(by),
                x_max=float(bx + bw), y_max=float(by + bh),
            )

            centroid_x = sum(p.x for p in valid_pts) / max(len(valid_pts), 1)
            centroid_y = sum(p.y for p in valid_pts) / max(len(valid_pts), 1)

            adjusted_confidence = min(0.99, confidence_score * (0.5 + 0.5 * c.score))

            results.append(DetectionResult(
                geometry=geometry,
                confidence=adjusted_confidence,
                class_name="roof_plane",
                metadata={
                    "polygon_vertices": [(p.x, p.y) for p in valid_pts],
                    "centroid": (centroid_x, centroid_y),
                    "source": c.source,
                    "area_pixels": c.area,
                    "solidity": round(c.solidity, 3),
                    "rectangularity": round(c.rectangularity, 3),
                    "contour_score": round(c.score, 3),
                },
            ))

        logger.info(
            f"RoofPlaneDetector completed. {len(results)} roof planes "
            f"(from {len(all_candidates)} scored, {len(unique_candidates)} unique)"
        )
        return results
