"""
This module defines the RoofCanvas widget, a QGraphicsView for displaying roof images
and enabling interactive geometry editing.
"""

from typing import Optional, List, Union
import numpy as np
import cv2 # For converting mask to QImage

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QSizePolicy,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPolygonItem
)
from PySide6.QtGui import QPixmap, QTransform, QMouseEvent, QWheelEvent, QPen, QBrush, QColor, QImage, QPolygonF
from PySide6.QtCore import Qt, Signal, QPointF, QRectF

from app.core.image.image_model import ImageInfo
from app.geometry.point import Point2D
from app.geometry.polygon import Polygon2D
from app.ai.segmentation_result import SegmentationResult
from app.ai.ai_result import DetectionResult, BoundingBox

class RoofCanvas(QGraphicsView):
    """
    A custom QGraphicsView for displaying roof images with zoom, pan, and
    interactive drawing capabilities for roof geometry.
    """

    # Signals for general canvas interactions
    mouse_pressed = Signal(QPointF, Qt.MouseButton) # Emits scene point and button
    mouse_moved = Signal(QPointF) # Emits current scene point
    mouse_released = Signal(QPointF, Qt.MouseButton) # Emits scene point and button
    zoom_level_changed = Signal(float)
    pan_offset_changed = Signal(QPointF)
    image_displayed = Signal(ImageInfo)
    image_cleared = Signal()

    # Signals for drawing mode interactions
    point_added_to_drawing = Signal(QPointF)
    point_moved_in_drawing = Signal(int, QPointF)
    polygon_drawing_finished = Signal()
    drawing_mode_changed = Signal(bool)

    def __init__(self, parent: QWidget = None):
        """
        Initializes the RoofCanvas.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setRenderHint(QGraphicsView.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag) # Initially no drag, will change for pan
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._last_pan_pos: QPointF = QPointF()
        self._zoom_factor: float = 1.0
        self._current_image_info: Optional[ImageInfo] = None

        # Drawing mode variables
        self._drawing_mode_active: bool = False
        self._current_drawing_points_items: List[QGraphicsEllipseItem] = []
        self._current_drawing_line_items: List[QGraphicsLineItem] = []
        self._rubber_band_line: Optional[QGraphicsLineItem] = None
        self._selected_point_item: Optional[QGraphicsEllipseItem] = None
        self._selected_point_index: int = -1
        self._is_dragging_point: bool = False

        # AI Overlay variables
        self._ai_overlay_active: bool = False
        self._ai_overlay_items: List[Union[QGraphicsPolygonItem, QGraphicsPixmapItem]] = []

        # Drawing styles
        self._point_pen = QPen(QColor(255, 0, 0), 2) # Red outline
        self._point_brush = QBrush(QColor(255, 0, 0, 150)) # Semi-transparent red fill
        self._selected_point_brush = QBrush(QColor(0, 255, 0, 150)) # Semi-transparent green fill
        self._line_pen = QPen(QColor(0, 0, 255), 2) # Blue line
        self._rubber_band_pen = QPen(QColor(255, 255, 0, 150), 1, Qt.PenStyle.DashLine) # Yellow dashed line

        # AI Overlay styles
        self._segmentation_mask_color = QColor(0, 255, 0, 80) # Semi-transparent green
        self._contour_pen = QPen(QColor(0, 255, 0), 2) # Green contour
        self._detection_box_pen = QPen(QColor(255, 165, 0), 2) # Orange box

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def current_image_info(self) -> Optional[ImageInfo]:
        """Returns the ImageInfo of the currently displayed image."""
        return self._current_image_info

    def set_drawing_mode(self, active: bool) -> None:
        """
        Activates or deactivates the interactive drawing mode.
        """
        if self._drawing_mode_active == active:
            return

        self._drawing_mode_active = active
        self.drawing_mode_changed.emit(active)

        if active:
            self.setDragMode(QGraphicsView.DragMode.NoDrag) # Disable panning while drawing
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.clear_drawing_visuals()
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag) # Reset drag mode
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            if self._rubber_band_line:
                self.scene.removeItem(self._rubber_band_line)
                self._rubber_band_line = None
            self.clear_drawing_visuals() # Clear visuals when exiting drawing mode

    def set_ai_overlay_mode(self, active: bool) -> None:
        """
        Activates or deactivates the AI overlay display.
        """
        if self._ai_overlay_active == active:
            return
        self._ai_overlay_active = active
        if not active:
            self.clear_ai_overlay_visuals()
        # If activating, the overlay will be drawn when display_ai_results is called

    def display_qpixmap(self, pixmap: QPixmap, image_info: ImageInfo) -> None:
        """
        Displays a QPixmap on the canvas and stores its information.
        """
        self.scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self._pixmap_item)
        self.scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_image_to_view()
        self._zoom_factor = self.transform().m11() # Update zoom factor based on fit
        self._current_image_info = image_info
        self.image_displayed.emit(image_info)
        self.clear_ai_overlay_visuals() # Clear old AI results when new image is loaded

    def clear_canvas(self) -> None:
        """
        Clears the canvas of all items and resets image info.
        """
        self.scene.clear()
        self._pixmap_item = None
        self._current_image_info = None
        self.clear_drawing_visuals()
        self.clear_ai_overlay_visuals()
        self.image_cleared.emit()

    def fit_image_to_view(self) -> None:
        """
        Fits the loaded image entirely within the view, maintaining aspect ratio.
        """
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11() # Update zoom factor based on fit
            self.zoom_level_changed.emit(self._zoom_factor)

    def update_drawing_visuals(self, points: List[Point2D]) -> None:
        """
        Updates the visual representation of the polygon being drawn on the canvas.
        """
        # Clear existing drawing items
        self.clear_drawing_visuals()

        # Draw points
        point_size = 8 / self._zoom_factor # Adjust point size based on zoom
        for i, p in enumerate(points):
            brush = self._point_brush
            if i == self._selected_point_index:
                brush = self._selected_point_brush
            point_item = QGraphicsEllipseItem(p.x - point_size/2, p.y - point_size/2, point_size, point_size)
            point_item.setPen(self._point_pen)
            point_item.setBrush(brush)
            self.scene.addItem(point_item)
            self._current_drawing_points_items.append(point_item)

        # Draw lines
        if len(points) > 1:
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i+1]
                line_item = QGraphicsLineItem(p1.x, p1.y, p2.x, p2.y)
                line_item.setPen(self._line_pen)
                self.scene.addItem(line_item)
                self._current_drawing_line_items.append(line_item)
            
            # Draw closing line if polygon has at least 3 points
            if len(points) >= 3:
                p_start = points[-1]
                p_end = points[0]
                closing_line_item = QGraphicsLineItem(p_start.x, p_start.y, p_end.x, p_end.y)
                closing_line_item.setPen(self._line_pen)
                self.scene.addItem(closing_line_item)
                self._current_drawing_line_items.append(closing_line_item)

        # Remove rubber band if it exists and drawing is complete
        if self._rubber_band_line and len(points) > 0 and not self._is_dragging_point:
             self.scene.removeItem(self._rubber_band_line)
             self._rubber_band_line = None

    def clear_drawing_visuals(self) -> None:
        """
        Removes all temporary drawing items (points, lines) from the scene.
        """
        for item in self._current_drawing_points_items:
            self.scene.removeItem(item)
        for item in self._current_drawing_line_items:
            self.scene.removeItem(item)
        if self._rubber_band_line:
            self.scene.removeItem(self._rubber_band_line)
            self._rubber_band_line = None
        self._current_drawing_points_items.clear()
        self._current_drawing_line_items.clear()
        self._selected_point_item = None
        self._selected_point_index = -1
        self._is_dragging_point = False

    def display_ai_results_overlay(self, ai_results: List[Union[DetectionResult, SegmentationResult]]) -> None:
        """
        Displays AI detection and segmentation results as an overlay on the canvas.
        """
        self.clear_ai_overlay_visuals() # Clear previous overlays

        if not self._ai_overlay_active:
            return

        for result in ai_results:
            if isinstance(result, SegmentationResult):
                if result.mask is not None and result.image_size is not None:
                    # Ensure mask is the same size as the original image displayed
                    mask_h, mask_w = result.mask.shape[:2]
                    if mask_w != result.image_size[0] or mask_h != result.image_size[1]:
                        # Resize mask to original image dimensions if necessary
                        mask_display = cv2.resize(result.mask, result.image_size, interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_display = result.mask

                    # Create a QImage from the mask
                    # Convert binary mask to 3-channel for QImage display
                    mask_colored = np.zeros((mask_display.shape[0], mask_display.shape[1], 4), dtype=np.uint8)
                    mask_colored[mask_display > 0] = [
                        self._segmentation_mask_color.red(),
                        self._segmentation_mask_color.green(),
                        self._segmentation_mask_color.blue(),
                        self._segmentation_mask_color.alpha()
                    ]
                    
                    q_mask_image = QImage(mask_colored.data, mask_colored.shape[1], mask_colored.shape[0], 
                                          mask_colored.shape[1] * 4, QImage.Format.Format_RGBA8888)
                    mask_pixmap = QPixmap.fromImage(q_mask_image)
                    mask_item = QGraphicsPixmapItem(mask_pixmap)
                    self.scene.addItem(mask_item)
                    self._ai_overlay_items.append(mask_item)

                    # Optionally, draw contours for segmentation
                    contours, _ = cv2.findContours(mask_display.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) >= 3:
                            polygon_points = [QPointF(p[0][0], p[0][1]) for p in contour.squeeze()]
                            q_polygon = QPolygonF(polygon_points)
                            polygon_item = QGraphicsPolygonItem(q_polygon)
                            polygon_item.setPen(self._contour_pen)
                            polygon_item.setBrush(Qt.NoBrush)
                            self.scene.addItem(polygon_item)
                            self._ai_overlay_items.append(polygon_item)

            elif isinstance(result, DetectionResult):
                bbox = result.bounding_box
                rect = QRectF(bbox.x_min, bbox.y_min, bbox.width, bbox.height)
                rect_item = self.scene.addRect(rect, self._detection_box_pen)
                self._ai_overlay_items.append(rect_item)

        # Ensure overlay items are above the image
        for item in self._ai_overlay_items:
            item.setZValue(1) # Set a higher Z-value than the pixmap item (default Z-value is 0)

    def clear_ai_overlay_visuals(self) -> None:
        """
        Removes all AI overlay items from the scene.
        """
        for item in self._ai_overlay_items:
            self.scene.removeItem(item)
        self._ai_overlay_items.clear()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel events for zooming.
        """
        if not self._pixmap_item: # Only zoom if an image is loaded
            super().wheelEvent(event)
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Save the current scene position under the mouse
        old_pos = self.mapToScene(event.position().toPoint())

        # Zoom
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
            self._zoom_factor *= zoom_in_factor
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
            self._zoom_factor *= zoom_out_factor

        # Get the new scene position under the mouse
        new_pos = self.mapToScene(event.position().toPoint())

        # Move scene to keep the old position under the mouse
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        self.zoom_level_changed.emit(self._zoom_factor)
        event.accept()

        # Update point sizes after zoom
        self.update_drawing_visuals(self._get_current_drawing_points_from_items())

    def _get_current_drawing_points_from_items(self) -> List[Point2D]:
        """Helper to get Point2D list from current drawing items."""
        return [Point2D(item.rect().center().x(), item.rect().center().y()) for item in self._current_drawing_points_items]

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse press events for panning or drawing.
        """
        scene_pos = self.mapToScene(event.position().toPoint())
        self.mouse_pressed.emit(scene_pos, event.button())

        if self._drawing_mode_active and self._pixmap_item:
            if event.button() == Qt.MouseButton.LeftButton:
                # Check if an existing point is clicked for dragging
                self._selected_point_item = None
                self._selected_point_index = -1
                for i, item in enumerate(self._current_drawing_points_items):
                    if item.contains(scene_pos):
                        self._selected_point_item = item
                        self._selected_point_index = i
                        self._is_dragging_point = True
                        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                        # Update visual to show selected point
                        self.update_drawing_visuals(self._get_current_drawing_points_from_items())
                        return # Don't add a new point if dragging existing
                
                # If no existing point is clicked, add a new one
                self.point_added_to_drawing.emit(scene_pos)
                self.viewport().setCursor(Qt.CursorShape.CrossCursor) # Keep cross cursor for adding
                self._is_dragging_point = False # Ensure not dragging
            elif event.button() == Qt.MouseButton.RightButton:
                self.polygon_drawing_finished.emit()
                self.set_drawing_mode(False) # Exit drawing mode after finishing
        elif self._pixmap_item: # Not in drawing mode, handle panning
            if event.button() == Qt.MouseButton.LeftButton:
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self._last_pan_pos = event.position()
        super().mousePressEvent(event) # Call super to ensure default behavior for other buttons

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse move events for panning or drawing.
        """
        scene_pos = self.mapToScene(event.position().toPoint())
        self.mouse_moved.emit(scene_pos)

        if self._drawing_mode_active and self._pixmap_item:
            if self._is_dragging_point and self._selected_point_index != -1:
                # Update the position of the dragged point
                self.point_moved_in_drawing.emit(self._selected_point_index, scene_pos)
            elif len(self._current_drawing_points_items) > 0 and not self._is_dragging_point:
                # Draw rubber band line from last point to current mouse position
                last_point_pos = self._current_drawing_points_items[-1].rect().center()
                if not self._rubber_band_line:
                    self._rubber_band_line = QGraphicsLineItem(last_point_pos.x(), last_point_pos.y(), scene_pos.x(), scene_pos.y())
                    self._rubber_band_line.setPen(self._rubber_band_pen)
                    self.scene.addItem(self._rubber_band_line)
                else:
                    self._rubber_band_line.setLine(last_point_pos.x(), last_point_pos.y(), scene_pos.x(), scene_pos.y())
        elif self._pixmap_item: # Not in drawing mode, handle panning
            if event.buttons() == Qt.MouseButton.LeftButton and self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
                delta = event.position() - self._last_pan_pos
                self._last_pan_pos = event.position()
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
                self.pan_offset_changed.emit(QPointF(delta.x(), delta.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse release events, ending panning or dragging.
        """
        scene_pos = self.mapToScene(event.position().toPoint())
        self.mouse_released.emit(scene_pos, event.button())

        if self._drawing_mode_active and self._pixmap_item:
            if event.button() == Qt.MouseButton.LeftButton:
                self._is_dragging_point = False
                self._selected_point_item = None
                self._selected_point_index = -1
                self.viewport().setCursor(Qt.CursorShape.CrossCursor) # Reset cursor
                # Update visuals to remove selected point highlight
                self.update_drawing_visuals(self._get_current_drawing_points_from_items())
        elif self._pixmap_item: # Not in drawing mode, handle panning
            if event.button() == Qt.MouseButton.LeftButton:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        """
        Handles resize events to re-fit the image if necessary.
        """
        super().resizeEvent(event)
        # Optionally re-fit image on resize, or just ensure it's centered
        # self.fit_image_to_view() # This might be too aggressive, only do if desired
        pass
