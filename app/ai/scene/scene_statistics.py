"""
Scene Statistics: Computes and logs statistics about detected roof scene.

Provides:
- Number of detected buildings and planes
- Average confidence and quality scores
- Average polygon vertices
- Processing time
- Quality distribution
"""

import logging
import time
from typing import List, Dict, Any
import numpy as np

from app.ai.config import get_detection_config
from app.core.logger import setup_logging

from .roof_scene import RoofScene
from .roof_group import RoofGroup
from .roof_plane import RoofPlane

logger: logging.Logger = setup_logging()


class SceneStatistics:
    """
    Computes and logs statistics about the detected roof scene.
    
    This runs after all processing stages and provides useful insights
    for monitoring pipeline performance and detection quality.
    
    Configuration from detection.yaml → scene_performance:
    - compute_statistics: Whether to compute statistics
    """
    
    def __init__(self):
        """Initialize the scene statistics computer."""
        self.config = get_detection_config()
        self.log_enabled = self.config.logging.log_scene_statistics
        if self.log_enabled:
            logger.info("SceneStatistics initialized")
    
    def compute_scene_statistics(self, scene: RoofScene, processing_time: float = 0.0) -> Dict[str, Any]:
        """
        Compute comprehensive statistics for a scene.
        
        Args:
            scene: RoofScene object
            processing_time: Total processing time in seconds
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'timestamp': scene.timestamp.isoformat(),
            'image_size': scene.image_size,
            'processing_time_ms': processing_time * 1000,
            'buildings': {
                'count': scene.total_buildings,
                'avg_confidence': scene.average_group_confidence,
                'avg_quality_score': scene.scene_quality_score,
            },
            'planes': {
                'count': scene.total_planes,
            },
            'groups': self._compute_group_statistics(scene.groups),
        }
        
        if self.log_enabled:
            self._log_statistics(stats)
        
        return stats
    
    def _compute_group_statistics(self, groups: List[RoofGroup]) -> Dict[str, Any]:
        """Compute statistics about roof groups."""
        if not groups:
            return {
                'count': 0,
                'avg_planes_per_group': 0.0,
                'avg_area_pixels': 0.0,
                'avg_confidence': 0.0,
                'avg_quality_score': 0.0,
                'confidence_range': {'min': 0.0, 'max': 0.0},
                'quality_distribution': {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0},
            }
        
        plane_counts = [g.plane_count for g in groups]
        areas = [g.total_area_pixels for g in groups]
        confidences = [g.average_confidence for g in groups]
        quality_scores = [g.group_quality_score for g in groups]
        
        # Quality level distribution
        quality_dist = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for group in groups:
            level = self._classify_quality_level(group.group_quality_score)
            if level in quality_dist:
                quality_dist[level] += 1
        
        return {
            'count': len(groups),
            'avg_planes_per_group': float(np.mean(plane_counts)) if plane_counts else 0.0,
            'min_planes_per_group': int(min(plane_counts)) if plane_counts else 0,
            'max_planes_per_group': int(max(plane_counts)) if plane_counts else 0,
            'avg_area_pixels': float(np.mean(areas)) if areas else 0.0,
            'total_area_pixels': float(sum(areas)) if areas else 0.0,
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'confidence_range': {
                'min': float(min(confidences)) if confidences else 0.0,
                'max': float(max(confidences)) if confidences else 0.0,
            },
            'avg_quality_score': float(np.mean(quality_scores)) if quality_scores else 0.0,
            'quality_range': {
                'min': float(min(quality_scores)) if quality_scores else 0.0,
                'max': float(max(quality_scores)) if quality_scores else 0.0,
            },
            'quality_distribution': quality_dist,
        }
    
    def _compute_plane_statistics(self, planes: List[RoofPlane]) -> Dict[str, Any]:
        """Compute statistics about roof planes."""
        if not planes:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'avg_quality_score': 0.0,
                'avg_polygon_vertices': 0,
                'area_range': {'min': 0.0, 'max': 0.0},
            }
        
        confidences = [p.confidence for p in planes]
        quality_scores = [p.quality_score for p in planes]
        vertex_counts = [p.polygon.corner_count if p.polygon else 4 for p in planes]
        areas = [p.area_pixels for p in planes]
        
        return {
            'count': len(planes),
            'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'confidence_range': {
                'min': float(min(confidences)) if confidences else 0.0,
                'max': float(max(confidences)) if confidences else 0.0,
            },
            'avg_quality_score': float(np.mean(quality_scores)) if quality_scores else 0.0,
            'quality_range': {
                'min': float(min(quality_scores)) if quality_scores else 0.0,
                'max': float(max(quality_scores)) if quality_scores else 0.0,
            },
            'avg_polygon_vertices': float(np.mean(vertex_counts)) if vertex_counts else 0.0,
            'vertex_range': {
                'min': int(min(vertex_counts)) if vertex_counts else 0,
                'max': int(max(vertex_counts)) if vertex_counts else 0,
            },
            'avg_area_pixels': float(np.mean(areas)) if areas else 0.0,
            'area_range': {
                'min': float(min(areas)) if areas else 0.0,
                'max': float(max(areas)) if areas else 0.0,
            },
        }
    
    def _classify_quality_level(self, score: float) -> str:
        """Classify quality score into levels."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _log_statistics(self, stats: Dict[str, Any]):
        """Log statistics in a structured format."""
        logger.info("=" * 60)
        logger.info("SCENE UNDERSTANDING STATISTICS")
        logger.info("=" * 60)
        
        logger.info(f"Processing time: {stats['processing_time_ms']:.1f} ms")
        logger.info(f"Image size: {stats['image_size']}")
        
        buildings = stats['buildings']
        logger.info(f"\nBUILDINGS:")
        logger.info(f"  Count: {buildings['count']}")
        logger.info(f"  Avg confidence: {buildings['avg_confidence']:.2f}")
        logger.info(f"  Avg quality score: {buildings['avg_quality_score']:.2f}")
        
        planes = stats['planes']
        logger.info(f"\nPLANES:")
        logger.info(f"  Total count: {planes['count']}")
        
        groups = stats['groups']
        logger.info(f"\nGROUPS:")
        logger.info(f"  Count: {groups['count']}")
        if groups['count'] > 0:
            logger.info(f"  Avg planes/group: {groups['avg_planes_per_group']:.1f}")
            logger.info(f"  Planes range: {groups['min_planes_per_group']}-{groups['max_planes_per_group']}")
            logger.info(f"  Avg area: {groups['avg_area_pixels']:.0f} pixels")
            logger.info(f"  Avg confidence: {groups['avg_confidence']:.2f}")
            logger.info(f"  Avg quality: {groups['avg_quality_score']:.2f}")
            
            qual_dist = groups['quality_distribution']
            logger.info(f"  Quality distribution:")
            logger.info(f"    Excellent: {qual_dist['excellent']}")
            logger.info(f"    Good: {qual_dist['good']}")
            logger.info(f"    Fair: {qual_dist['fair']}")
            logger.info(f"    Poor: {qual_dist['poor']}")
        
        logger.info("=" * 60)
    
    def create_summary_report(self, scene: RoofScene, processing_time: float = 0.0) -> str:
        """
        Create a human-readable summary report of the scene.
        
        Args:
            scene: RoofScene object
            processing_time: Total processing time in seconds
        
        Returns:
            Formatted summary string
        """
        stats = self.compute_scene_statistics(scene, processing_time)
        
        report = []
        report.append("ROOF DETECTION SCENE SUMMARY")
        report.append("=" * 50)
        report.append(f"Processing time: {stats['processing_time_ms']:.1f} ms")
        report.append(f"Image size: {stats['image_size']}")
        report.append("")
        
        report.append("DETECTION RESULTS:")
        report.append(f"  Buildings detected: {stats['buildings']['count']}")
        report.append(f"  Total planes: {stats['planes']['count']}")
        
        if stats['buildings']['count'] > 0:
            report.append("")
            report.append("QUALITY METRICS:")
            report.append(f"  Avg building confidence: {stats['buildings']['avg_confidence']:.2%}")
            report.append(f"  Avg scene quality: {stats['buildings']['avg_quality_score']:.2%}")
        
        if stats['groups']['count'] > 0:
            report.append("")
            report.append("GROUP STATISTICS:")
            report.append(f"  Planes per building: {stats['groups']['avg_planes_per_group']:.1f} (avg)")
            report.append(f"  Quality distribution:")
            qual = stats['groups']['quality_distribution']
            total = sum(qual.values())
            if total > 0:
                report.append(f"    Excellent: {qual['excellent']} ({qual['excellent']/total*100:.0f}%)")
                report.append(f"    Good: {qual['good']} ({qual['good']/total*100:.0f}%)")
                report.append(f"    Fair: {qual['fair']} ({qual['fair']/total*100:.0f}%)")
                report.append(f"    Poor: {qual['poor']} ({qual['poor']/total*100:.0f}%)")
        
        report.append("=" * 50)
        
        return "\n".join(report)
