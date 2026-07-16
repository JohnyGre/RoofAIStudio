"""
Roof post-processing module.
Cleans raw sub-detector predictions, merges overlaps, extracts shared boundary edges
(polygon intersection) as RoofEdge objects, and structures roof features.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import uuid
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, LineString as ShapelyLineString, MultiLineString as ShapelyMultiLineString
from shapely.ops import unary_union

from app.ai.ai_result import DetectionResult, PolygonGeometry, BaseGeometry, BBoxGeometry
from app.geometry.point import Point2D
from app.geometry.edge import RoofEdge

@dataclass
class RoofFeature:
    """
    Represents a structured roof feature (e.g., opening, chimney, skylight)
    enriched with semantic relationships and geometric properties.
    """
    id: uuid.UUID
    feature_type: str  # e.g., "opening", "chimney", "skylight"
    geometry: BaseGeometry
    confidence: float
    parent_plane_id: Optional[uuid.UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PostProcessedResults:
    """
    Structured output of the RoofPostProcessor containing cleaned planes,
    relationships, extracted edges, and features.
    """
    planes: List[DetectionResult]
    edges: List[RoofEdge]
    features: List[RoofFeature]

class RoofPostProcessor:
    """
    Post-processes raw detection results from all sub-detectors.
    Merges duplicate planes, performs polygon intersection to find shared boundaries
    (ridges/valleys), maps features (openings) to their parent planes, and formats
    the final topology.
    """

    def __init__(self, adjacency_threshold_pixels: float = 8.0, overlap_merge_threshold: float = 0.85):
        """
        Args:
            adjacency_threshold_pixels: Maximum distance between two polygons to consider them adjacent.
            overlap_merge_threshold: Intersection over Union (IoU) threshold to merge overlapping planes.
        """
        self.adjacency_threshold = adjacency_threshold_pixels
        self.overlap_merge_threshold = overlap_merge_threshold

    def post_process(self, raw_results: List[DetectionResult]) -> PostProcessedResults:
        """
        Main post-processing entry point.
        """
        # Separate results by class_name
        plane_results = [r for r in raw_results if r.class_name in ("roof_plane", "roof_area")]
        feature_results = [r for r in raw_results if r.class_name in ("opening", "potential_opening", "chimney", "skylight")]

        # 1. Clean and merge overlapping planes
        cleaned_planes = self._merge_overlapping_planes(plane_results)

        # 2. Extract shared boundary edges (ridges/valleys/hips) via polygon boundary intersection
        extracted_edges = self._extract_shared_edges(cleaned_planes)

        # 3. Associate and structure features (openings, chimneys)
        structured_features = self._structure_features(feature_results, cleaned_planes)

        return PostProcessedResults(
            planes=cleaned_planes,
            edges=extracted_edges,
            features=structured_features
        )

    def _merge_overlapping_planes(self, planes: List[DetectionResult]) -> List[DetectionResult]:
        """
        Merges duplicate or highly overlapping plane detections using Shapely.
        """
        if not planes:
            return []

        # Sort planes by confidence/area to process larger/higher confidence first
        sorted_planes = sorted(planes, key=lambda p: (p.confidence, p.bounding_box.area), reverse=True)
        merged: List[DetectionResult] = []

        for p in sorted_planes:
            if not isinstance(p.geometry, PolygonGeometry):
                continue
            
            # Check if this plane overlaps significantly with any already-merged plane
            p_shape = ShapelyPolygon(p.geometry.vertices)
            if not p_shape.is_valid:
                p_shape = p_shape.buffer(0)  # Try to fix invalid polygon

            should_merge = False
            for idx, existing in enumerate(merged):
                existing_shape = ShapelyPolygon(existing.geometry.vertices)
                if not existing_shape.is_valid:
                    existing_shape = existing_shape.buffer(0)

                intersection_area = p_shape.intersection(existing_shape).area
                union_area = p_shape.union(existing_shape).area
                iou = intersection_area / union_area if union_area > 0 else 0.0

                if iou > self.overlap_merge_threshold:
                    # Overlap is very high, merge by keeping the one with higher confidence
                    should_merge = True
                    break

            if not should_merge:
                merged.append(p)

        return merged

    def _extract_shared_edges(self, planes: List[DetectionResult]) -> List[RoofEdge]:
        """
        Finds adjacent planes and extracts their shared boundaries as RoofEdge objects.
        This replaces Hough-transform edge estimation.
        """
        edges: List[RoofEdge] = []
        num_planes = len(planes)
        if num_planes < 2:
            return edges

        shapes = [ShapelyPolygon(p.geometry.vertices) for p in planes]
        # Ensure all polygons are valid
        shapes = [s if s.is_valid else s.buffer(0) for s in shapes]

        for i in range(num_planes):
            for j in range(i + 1, num_planes):
                p1, p2 = planes[i], planes[j]
                s1, s2 = shapes[i], shapes[j]

                # Buffering slightly to handle small gaps/noises in prediction boundaries
                # A buffer of threshold/2 on each side allows intersection detection
                buffered_s1 = s1.buffer(self.adjacency_threshold / 2.0)
                buffered_s2 = s2.buffer(self.adjacency_threshold / 2.0)

                overlap = buffered_s1.intersection(buffered_s2)
                if overlap.is_empty:
                    continue

                # If overlap is a polygon or linestring, extract the centerline/shared boundary
                # To get the exact 2D boundary segment shared between s1 and s2:
                # We intersect the original boundaries with the buffered overlap
                shared_line = s1.boundary.intersection(s2.buffer(self.adjacency_threshold))
                
                if shared_line.is_empty:
                    continue

                # Extract line segments from the intersection result
                lines: List[ShapelyLineString] = []
                if isinstance(shared_line, ShapelyLineString):
                    lines.append(shared_line)
                elif isinstance(shared_line, (ShapelyMultiLineString, any)) and hasattr(shared_line, "geoms"):
                    for g in shared_line.geoms:
                        if isinstance(g, ShapelyLineString):
                            lines.append(g)

                for line in lines:
                    if line.length < 5.0:  # Ignore very short segments
                        continue
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        start_pt = Point2D(coords[0][0], coords[0][1])
                        end_pt = Point2D(coords[-1][0], coords[-1][1])
                        
                        # Determine edge type:
                        # For shared inner edges between planes, it's typically a ridge, valley, or hip.
                        # For classical 2D pipelines, we classify it as candidate "ridge" if horizontal-ish, or "hip/valley".
                        edge_type = "ridge"  # Candidate default type for shared edges
                        
                        edges.append(RoofEdge(
                            start_point=start_pt,
                            end_point=end_pt,
                            left_plane_id=str(p1.id),
                            right_plane_id=str(p2.id),
                            edge_type=edge_type
                        ))

        return edges

    def _structure_features(self, features: List[DetectionResult], planes: List[DetectionResult]) -> List[RoofFeature]:
        """
        Converts generic feature detections to structured RoofFeatures
        and associates each feature with its parent roof plane.
        """
        structured: List[RoofFeature] = []
        if not features:
            return structured

        plane_shapes = [(p.id, ShapelyPolygon(p.geometry.vertices)) for p in planes if isinstance(p.geometry, PolygonGeometry)]

        for f in features:
            f_geometry = f.geometry
            f_type = f.class_name
            
            # Find parent plane by looking at which plane polygon contains the feature's centroid
            parent_id = None
            if isinstance(f_geometry, PolygonGeometry) and len(f_geometry.vertices) > 0:
                f_shape = ShapelyPolygon(f_geometry.vertices)
                centroid = f_shape.centroid
                for p_id, p_shape in plane_shapes:
                    if p_shape.contains(centroid):
                        parent_id = p_id
                        break
            elif isinstance(f_geometry, BBoxGeometry):
                # Fallback to bbox center
                cx = (f_geometry.x_min + f_geometry.x_max) / 2.0
                cy = (f_geometry.y_min + f_geometry.y_max) / 2.0
                from shapely.geometry import Point as ShapelyPoint
                pt = ShapelyPoint(cx, cy)
                for p_id, p_shape in plane_shapes:
                    if p_shape.contains(pt):
                        parent_id = p_id
                        break

            structured.append(RoofFeature(
                id=f.id,
                feature_type=f_type,
                geometry=f_geometry,
                confidence=f.confidence,
                parent_plane_id=parent_id,
                metadata=f.metadata
            ))

        return structured
