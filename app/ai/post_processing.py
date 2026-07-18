"""
This module defines post-processing steps for raw AI detection results.
It aims to refine detected planes and extract topological features like shared edges.
"""

from dataclasses import dataclass
from typing import List, Union, Optional

from app.ai.ai_result import DetectionResult, SegmentationResult, GeometryPredictionResult
from app.geometry.edge import Edge # Assuming Edge is a domain model for edges
from app.geometry.point import Point2D # Assuming Point2D is a domain model for points

@dataclass
class PostProcessedAIResults:
    """
    Data structure to hold AI results after post-processing,
    including refined planes and extracted edges.
    """
    planes: List[DetectionResult]
    edges: List[Edge] # Shared edges between planes, or other significant lines
    # Add other refined features as needed

class RoofPostProcessor:
    """
    Performs post-processing on raw AI detection results to refine them
    and extract higher-level geometric features.
    """
    def __init__(self):
        pass

    def post_process(self, raw_ai_results: List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]) -> PostProcessedAIResults:
        """
        Refines raw AI results. For now, it just passes through detection results
        and extracts dummy edges.
        """
        refined_planes: List[DetectionResult] = []
        extracted_edges: List[Edge] = []

        for res in raw_ai_results:
            if isinstance(res, DetectionResult) and res.class_name == "roof_area":
                # For now, just pass through the detected roof areas as refined planes
                refined_planes.append(res)
                
                # Dummy edge extraction for demonstration
                bbox = res.bounding_box
                p1 = Point2D(bbox.x_min, bbox.y_min)
                p2 = Point2D(bbox.x_max, bbox.y_min)
                p3 = Point2D(bbox.x_max, bbox.y_max)
                p4 = Point2D(bbox.x_min, bbox.y_max)
                
                # Create dummy edges from bounding box
                extracted_edges.append(Edge(start_point=Point2D(p1.x, p1.y), end_point=Point2D(p2.x, p2.y)))
                extracted_edges.append(Edge(start_point=Point2D(p2.x, p2.y), end_point=Point2D(p3.x, p3.y)))
                extracted_edges.append(Edge(start_point=Point2D(p3.x, p3.y), end_point=Point2D(p4.x, p4.y)))
                extracted_edges.append(Edge(start_point=Point2D(p4.x, p4.y), end_point=Point2D(p1.x, p1.y)))

        return PostProcessedAIResults(planes=refined_planes, edges=extracted_edges)
