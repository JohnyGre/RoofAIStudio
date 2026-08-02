"""
Roof Overlay Visualization: Renders RoofScene with improved visualization.

Displays:
- Group IDs
- Quality scores
- Plane counts
- Bounding boxes
- Polygons (if available)

Replaces raw YOLO labels with meaningful scene information.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import cv2

from app.ai.config import get_detection_config
from app.core.logger import setup_logging
from app.ai.scene import RoofScene, RoofGroup

logger: logging.Logger = setup_logging()


class RoofOverlayRenderer:
    """
    Renders roof detection results on images for visualization.
    
    Features:
    - Group IDs and metadata
    - Quality scores with color coding
    - Bounding boxes and polygons
    - Statistics overlay
    - Clean, production-ready visualization
    """
    
    # Color palette for quality scores
    QUALITY_COLORS = {
        'excellent': (0, 255, 0),      # Green
        'good': (0, 255, 255),         # Yellow
        'fair': (0, 165, 255),         # Orange
        'poor': (0, 0, 255),           # Red
    }
    
    # Default colors
    COLOR_BORDER = (200, 200, 200)     # Light gray for borders
    COLOR_TEXT = (255, 255, 255)       # White for text
    COLOR_BACKGROUND = (0, 0, 0)       # Black for text background
    
    def __init__(self):
        """Initialize the overlay renderer."""
        self.config = get_detection_config()
        self.log_enabled = self.config.logging.log_scene_statistics
    
    def render_scene(
        self,
        image: np.ndarray,
        scene: RoofScene,
        show_polygons: bool = True,
        show_bounding_boxes: bool = True,
        show_statistics: bool = True,
        show_quality_score: bool = True,
    ) -> np.ndarray:
        """
        Render scene on image.
        
        Args:
            image: Input image (BGR format)
            scene: RoofScene with detected groups
            show_polygons: Whether to render polygon outlines
            show_bounding_boxes: Whether to render bounding boxes
            show_statistics: Whether to show text labels
            show_quality_score: Whether to color by quality
        
        Returns:
            Image with overlays rendered
        """
        result = image.copy()
        
        if not scene.groups:
            logger.debug("Scene is empty, returning image unchanged")
            return result
        
        # Render each group
        for group in scene.groups:
            self._render_group(
                result,
                group,
                show_polygons=show_polygons,
                show_bounding_boxes=show_bounding_boxes,
                show_statistics=show_statistics,
                show_quality_score=show_quality_score,
            )
        
        # Render scene statistics if requested
        if show_statistics:
            self._render_scene_statistics(result, scene)
        
        return result
    
    def _render_group(
        self,
        image: np.ndarray,
        group: RoofGroup,
        show_polygons: bool = True,
        show_bounding_boxes: bool = True,
        show_statistics: bool = True,
        show_quality_score: bool = True,
    ):
        """Render a single group on the image."""
        if not group.planes:
            return
        
        # Determine color based on quality
        if show_quality_score:
            quality_level = self._classify_quality(group.group_quality_score)
            color = self.QUALITY_COLORS.get(quality_level, self.COLOR_BORDER)
        else:
            color = self.COLOR_BORDER
        
        # Render bounding box
        if show_bounding_boxes:
            self._render_bounding_box(image, group, color)
        
        # Render polygons for each plane
        if show_polygons:
            for plane in group.planes:
                if plane.polygon and plane.polygon.vertices:
                    self._render_polygon(image, plane.polygon, color)
        
        # Render text labels
        if show_statistics:
            self._render_group_labels(image, group, color)
    
    def _render_bounding_box(
        self,
        image: np.ndarray,
        group: RoofGroup,
        color: Tuple[int, int, int],
    ):
        """Render bounding box for a group."""
        x_min, y_min, x_max, y_max = group.bounding_box
        
        x_min, y_min = int(x_min), int(y_min)
        x_max, y_max = int(x_max), int(y_max)
        
        # Clamp to image bounds
        height, width = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness=2)
    
    def _render_polygon(
        self,
        image: np.ndarray,
        polygon,
        color: Tuple[int, int, int],
    ):
        """Render polygon outline for a plane."""
        if not polygon.vertices or len(polygon.vertices) < 3:
            return
        
        # Convert vertices to numpy array of integers
        vertices = np.array(polygon.vertices, dtype=np.int32)
        
        # Reshape for cv2.polylines
        vertices = vertices.reshape((-1, 1, 2))
        
        # Draw polygon
        cv2.polylines(image, [vertices], isClosed=True, color=color, thickness=2)
    
    def _render_group_labels(
        self,
        image: np.ndarray,
        group: RoofGroup,
        color: Tuple[int, int, int],
    ):
        """Render text labels for a group."""
        if not group.planes:
            return
        
        # Position labels at group center or top of bounding box
        x_min, y_min, x_max, y_max = group.bounding_box
        label_x = int((x_min + x_max) / 2)
        label_y = max(10, int(y_min) - 5)
        
        # Prepare labels
        labels = [
            f"ID: {group.id}",
            f"Planes: {group.plane_count}",
            f"Quality: {group.group_quality_score:.1%}",
            f"Confidence: {group.average_confidence:.1%}",
        ]
        
        # Render labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 15
        
        for i, label in enumerate(labels):
            y_pos = label_y + (i * line_height)
            
            # Get text size for background
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Draw background rectangle
            cv2.rectangle(
                image,
                (label_x - 2, y_pos - text_size[1] - 2),
                (label_x + text_size[0] + 2, y_pos + 2),
                self.COLOR_BACKGROUND,
                thickness=-1,  # Filled
            )
            
            # Draw text
            cv2.putText(
                image,
                label,
                (label_x, y_pos),
                font,
                font_scale,
                self.COLOR_TEXT,
                thickness,
            )
    
    def _render_scene_statistics(self, image: np.ndarray, scene: RoofScene):
        """Render scene-level statistics at the bottom of the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        margin = 10
        
        # Prepare statistics
        stats = [
            f"Buildings: {scene.total_buildings}",
            f"Planes: {scene.total_planes}",
            f"Avg Quality: {scene.scene_quality_score:.1%}",
            f"Avg Confidence: {scene.average_group_confidence:.1%}",
        ]
        
        # Draw at bottom-left
        height = image.shape[0]
        start_y = height - (len(stats) * line_height) - margin
        
        for i, stat in enumerate(stats):
            y_pos = start_y + (i * line_height)
            
            # Get text size
            text_size = cv2.getTextSize(stat, font, font_scale, thickness)[0]
            
            # Draw background
            cv2.rectangle(
                image,
                (margin - 2, y_pos - text_size[1] - 2),
                (margin + text_size[0] + 2, y_pos + 2),
                self.COLOR_BACKGROUND,
                thickness=-1,
            )
            
            # Draw text
            cv2.putText(
                image,
                stat,
                (margin, y_pos),
                font,
                font_scale,
                self.COLOR_TEXT,
                thickness,
            )
    
    def _classify_quality(self, score: float) -> str:
        """Classify quality score into levels."""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
