"""
Scene Understanding Module

Transforms raw YOLO detections into intelligent roof scene representations.

Pipeline:
  YOLO Detection → Instance Filtering → Roof Grouping → Geometry Refinement → Quality Scoring → RoofScene

Components:
  - RoofPlane: Individual roof plane (mask, polygon, confidence)
  - RoofGroup: Group of related roof planes (building, with metadata)
  - RoofScene: Complete scene containing all groups and future features
  - Services: InstanceFilter, RoofGrouper, PolygonRefiner, QualityScorer, SceneStatistics
  - Utils: PolygonCache for performance optimization
"""

from .roof_plane import RoofPlane, RoofPlanePolygon
from .roof_group import RoofGroup
from .roof_scene import RoofScene
from .instance_filter import InstanceFilter
from .roof_grouper import RoofGrouper
from .polygon_refiner import PolygonRefiner
from .quality_scorer import QualityScorer, QualityMetrics
from .scene_statistics import SceneStatistics
from .polygon_cache import PolygonCache

__all__ = [
    'RoofPlane',
    'RoofPlanePolygon',
    'RoofGroup',
    'RoofScene',
    'InstanceFilter',
    'RoofGrouper',
    'PolygonRefiner',
    'QualityScorer',
    'QualityMetrics',
    'SceneStatistics',
    'PolygonCache',
]
