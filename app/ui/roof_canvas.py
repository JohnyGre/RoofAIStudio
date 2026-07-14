"""
This module defines the RoofCanvas widget, a QGraphicsView for displaying roof images
and enabling interactive geometry editing.
"""

from typing import Optional, List, Union
import numpy as np
import cv2 # For converting mask to QImage

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QSizePolicy,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem, QGraphicsItem
)
from PySide6.QtGui import QPixmap, QTransform, QMouseEvent, QWheelEvent, QPen, QBrush, QColor, QImage, QPolygonF, QPainter
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from app.core.logger import setup_logging # Import logger

from app.core.image.image_model import ImageInfo
from app.geometry.point import Point2D
from app.geometry.polygon import Polygon2D
from app.ai.segmentation_result import SegmentationResult
from app.ai.ai_result import DetectionResult, BoundingBox

logger = setup_logging() # Initialize logger

class DraggableVertexItem(QGraphicsEllipseItem):
    """
    Small draggable vertex item used for AI overlay. Stores an index and notifies
    parent RoofCanvas on position changes via the canvas signal.
    """
    def __init__(self, index: int, pos: QPointF, size: float, canvas: 'RoofCanvas'):
        # Create ellipse centered at (0,0); use setPos to place it at scene coords
        super().__init__(-size/2, -size/2, size, size)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self._index = index
        self._canvas = canvas
        # Visual style
        self.setPen(canvas._point_pen)
        self.setBrush(canvas._point_brush)
        # Suppress emits during initial placement
        self._suppress_move_emit = True
        # Place at scene position
        self.setPos(pos)

    def enable_move_emits(self) -> None:
        """Enable emitting move events (call after adding items to scene)."""
        self._suppress_move_emit = False

    def itemChange(self, change, value):
        from PySide6.QtWidgets import QGraphicsItem
        if change == QGraphicsItem.ItemPositionHasChanged:
            try:
                # Only emit if not suppressed (i.e., user interaction)
                if not getattr(self, '_suppress_move_emit', False):
                    self._canvas.ai_overlay_vertex_moved.emit(self._index, self.scenePos())
            except Exception:
                pass
        return super().itemChange(change, value)


class RoofCanvas(QGraphicsView):
    """
    A custom QGraphicsView for displaying roof images with zoom, pan, and
    interactive drawing capabilities for roof geometry and calibration.
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

    # New signal for AI overlay vertex moves: emits (index, scene QPointF)
    ai_overlay_vertex_moved = Signal(int, QPointF)
    # Debounced version (emitted after short pause) to reduce update frequency during dragging
    ai_overlay_vertex_moved_debounced = Signal(int, QPointF)
    # Signal emitted when a plane overlay is selected: (index, list_of_pixel_tuples)
    ai_overlay_plane_selected = Signal(int, list)

    # New signal for calibration
    calibration_points_selected = Signal(Point2D, Point2D) # Emits two pixel points

    def __init__(self, parent: QWidget = None):
        """
        Initializes the RoofCanvas.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
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

        # Calibration mode variables
        self._calibration_mode_active: bool = False
        self._calibration_points: List[QPointF] = []
        self._calibration_point_items: List[QGraphicsEllipseItem] = []
        self._calibration_line_item: Optional[QGraphicsLineItem] = None

        # AI Overlay variables
        self._ai_overlay_active: bool = False
        self._ai_overlay_items: List[Union[QGraphicsPolygonItem, QGraphicsPixmapItem, QGraphicsRectItem]] = [] # Added QGraphicsRectItem
        # Support multiple detected planes: list of dicts {polygon_item, polygon_points, color}
        self._ai_planes: List[dict] = []
        self._selected_ai_plane_index: int = -1
        # Polygon overlay and draggable vertices for the SELECTED plane
        self._ai_polygon_item: Optional[QGraphicsPolygonItem] = None
        self._ai_vertex_items: List[DraggableVertexItem] = []
        self._ai_vertex_label_items: List[QGraphicsSimpleTextItem] = []
        self._current_ai_polygon_points: List[QPointF] = []
        # Area text item for AI overlay (shared, shows area for selected plane)
        self._ai_area_text_item: Optional[QGraphicsSimpleTextItem] = None
        # Debounce infrastructure for vertex moved events
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_interval_ms = 100
        self._debounce_timer.timeout.connect(self._flush_debounced_moves)
        self._debounce_pending: dict = {}
        # Color palette for planes
        self._ai_plane_colors = [QColor(0, 170, 0), QColor(0, 85, 170), QColor(170, 85, 0), QColor(170, 0, 170), QColor(85,170,85)]
        # Connect internal overlay move signal to handler
        self.ai_overlay_vertex_moved.connect(self._on_ai_overlay_vertex_moved)

        # Drawing styles
        self._point_pen = QPen(QColor(255, 0, 0), 2)
        self._point_brush = QBrush(QColor(255, 0, 0, 150))
        self._selected_point_brush = QBrush(QColor(0, 255, 0, 150))
        self._line_pen = QPen(QColor(0, 0, 255), 2)
        self._rubber_band_pen = QPen(QColor(255, 255, 0, 150), 1, Qt.PenStyle.DashLine)

        # Calibration styles
        self._calibration_point_pen = QPen(QColor(255, 255, 0), 2) # Yellow outline
        self._calibration_point_brush = QBrush(QColor(255, 255, 0, 150)) # Semi-transparent yellow fill
        self._calibration_line_pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.DotLine) # Yellow dotted line

        # AI Overlay styles
        self._segmentation_mask_color = QColor(0, 255, 0, 80)
        self._contour_pen = QPen(QColor(0, 255, 0), 2)
        self._detection_box_pen = QPen(QColor(255, 165, 0), 2)

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
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.clear_drawing_visuals()
            self.set_calibration_mode(False) # Deactivate calibration mode if drawing starts
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            if self._rubber_band_line:
                self.scene.removeItem(self._rubber_band_line)
                self._rubber_band_line = None
            self.clear_drawing_visuals()

    def set_calibration_mode(self, active: bool) -> None:
        """
        Activates or deactivates the interactive calibration mode.
        """
        if self._calibration_mode_active == active:
            return
        
        self._calibration_mode_active = active
        if active:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.clear_drawing_visuals() # Clear drawing visuals if calibration starts
            self.clear_calibration_visuals() # Clear previous calibration points
            self._calibration_points.clear()
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            self.clear_calibration_visuals()

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
        # Reset selection when re-enabling
        if active:
            self._selected_ai_plane_index = -1

    def display_qpixmap(self, pixmap: QPixmap, image_info: ImageInfo) -> None:
        """
        Displays a QPixmap on the canvas and stores its information.
        """
        self.scene.clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self._pixmap_item)
        self.scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fit_image_to_view()
        self._zoom_factor = self.transform().m11()
        self._current_image_info = image_info
        self.image_displayed.emit(image_info)
        self.clear_ai_overlay_visuals()
        self.clear_drawing_visuals()
        self.clear_calibration_visuals()

    def clear_canvas(self) -> None:
        """
        Clears the canvas of all items and resets image info.
        """
        self.scene.clear()
        self._pixmap_item = None
        self._current_image_info = None
        self.clear_drawing_visuals()
        self.clear_calibration_visuals()
        self.clear_ai_overlay_visuals()
        self.image_cleared.emit()

    def fit_image_to_view(self) -> None:
        """
        Fits the loaded image entirely within the view, maintaining aspect ratio.
        """
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
            self.zoom_level_changed.emit(self._zoom_factor)

    def update_drawing_visuals(self, points: List[Point2D]) -> None:
        """
        Updates the visual representation of the polygon being drawn on the canvas.
        """
        # Clear existing drawing items
        self.clear_drawing_visuals()

        # Draw points
        point_size = 8 / self._zoom_factor
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

    def update_calibration_visuals(self) -> None:
        """
        Updates the visual representation of calibration points and line.
        """
        self.clear_calibration_visuals()
        if not self._calibration_mode_active or not self._calibration_points:
            return

        point_size = 8 / self._zoom_factor
        for i, p in enumerate(self._calibration_points):
            point_item = QGraphicsEllipseItem(p.x() - point_size/2, p.y() - point_size/2, point_size, point_size)
            point_item.setPen(self._calibration_point_pen)
            point_item.setBrush(self._calibration_point_brush)
            self.scene.addItem(point_item)
            self._calibration_point_items.append(point_item)

        if len(self._calibration_points) == 2:
            p1 = self._calibration_points[0]
            p2 = self._calibration_points[1]
            line_item = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line_item.setPen(self._calibration_line_pen)
            self.scene.addItem(line_item)
            self._calibration_line_item = line_item

    def update_ai_overlay_visuals(self) -> None:
        """
        Updates vertex marker sizes and label positions according to current zoom level.
        Also repositions area text if present.
        """
        if not self._ai_polygon_item or not self._ai_vertex_items:
            # Still update area position relative to polygon centroid if area item exists
            if self._ai_area_text_item and self._current_ai_polygon_points:
                centroid = QPointF(0.0, 0.0)
                for p in self._current_ai_polygon_points:
                    centroid += p
                centroid /= max(1, len(self._current_ai_polygon_points))
                self._ai_area_text_item.setPos(centroid + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))
            return
        point_size = max(4.0, 8.0 / max(0.1, self._zoom_factor))
        for v in self._ai_vertex_items:
            # keep center at same scene position, adjust rect to new size
            scene_pos = v.scenePos()
            v.setRect(-point_size/2, -point_size/2, point_size, point_size)
            v.setPos(scene_pos)
        for i, label in enumerate(self._ai_vertex_label_items):
            if i < len(self._ai_vertex_items):
                v = self._ai_vertex_items[i]
                scene_pos = v.scenePos()
                label.setPos(scene_pos + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))
        # Reposition area text to polygon centroid if present
        if self._ai_area_text_item and self._current_ai_polygon_points:
            centroid = QPointF(0.0, 0.0)
            for p in self._current_ai_polygon_points:
                centroid += p
            centroid /= max(1, len(self._current_ai_polygon_points))
            self._ai_area_text_item.setPos(centroid + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))

    def _on_ai_overlay_vertex_moved(self, index: int, scene_pos: QPointF) -> None:
        """
        Internal handler when a vertex is moved by user. Update polygon visual immediately.
        Also schedule a debounced emission for external consumers.
        """
        if not (0 <= index < len(self._current_ai_polygon_points)):
            return
        # Update point immediately for visual feedback
        self._current_ai_polygon_points[index] = scene_pos
        if self._ai_polygon_item:
            try:
                new_poly = QPolygonF(self._current_ai_polygon_points)
                self._ai_polygon_item.setPolygon(new_poly)
            except Exception:
                pass
        # Update label position
        if index < len(self._ai_vertex_label_items):
            label = self._ai_vertex_label_items[index]
            label.setPos(scene_pos + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))

        # Store pending move for debounce and restart timer
        try:
            self._debounce_pending[int(index)] = scene_pos
            self._debounce_timer.start(self._debounce_interval_ms)
        except Exception:
            pass

    def clear_calibration_visuals(self) -> None:
        """
        Removes all calibration drawing items from the scene.
        """
        for item in self._calibration_point_items:
            self.scene.removeItem(item)
        if self._calibration_line_item:
            self.scene.removeItem(self._calibration_line_item)
        self._calibration_point_items.clear()
        self._calibration_line_item = None

    def display_ai_results_overlay(self, ai_results: List[Union[DetectionResult, SegmentationResult]]) -> None:
        """
        Displays AI detection and segmentation results as an overlay on the canvas.
        Supports multiple detected planes; the first plane is selected by default.
        """
        self.clear_ai_overlay_visuals()

        if not self._ai_overlay_active:
            return

        plane_index = 0
        for result in ai_results:
            if isinstance(result, SegmentationResult):
                if result.mask is not None and result.image_size is not None:
                    mask_h, mask_w = result.mask.shape[:2]
                    if mask_w != result.image_size[0] or mask_h != result.image_size[1]:
                        mask_display = cv2.resize(result.mask, result.image_size, interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_display = result.mask

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
                # If polygon vertices present in metadata, use polygon overlay
                polygon_pts = None
                if isinstance(result.metadata, dict):
                    polygon_pts = result.metadata.get("polygon_vertices") or result.metadata.get("contour_polygon")

                if polygon_pts:
                    qpoints = [QPointF(float(x), float(y)) for (x, y) in polygon_pts]
                    qpoly = QPolygonF(qpoints)
                    polygon_item = QGraphicsPolygonItem(qpoly)

                    # Assign color from palette
                    color = self._ai_plane_colors[plane_index % len(self._ai_plane_colors)]
                    pen = QPen(color, 2)
                    polygon_item.setPen(pen)
                    polygon_item.setBrush(Qt.NoBrush)

                    self.scene.addItem(polygon_item)
                    polygon_item.setZValue(1)
                    self._ai_overlay_items.append(polygon_item)

                    # Store plane info
                    self._ai_planes.append({
                        'polygon_item': polygon_item,
                        'polygon_points': qpoints,
                        'color': color
                    })

                    # Select the first plane by default
                    if self._selected_ai_plane_index == -1:
                        self._select_ai_plane(plane_index)

                    plane_index += 1
                else:
                    rect = QRectF(bbox.x_min, bbox.y_min, bbox.width, bbox.height)
                    rect_item = self.scene.addRect(rect, self._detection_box_pen)
                    self._ai_overlay_items.append(rect_item)

        # Ensure area text is above polygons
        if self._ai_area_text_item:
            self._ai_area_text_item.setZValue(4)
            if self._current_ai_polygon_points:
                centroid = QPointF(0.0, 0.0)
                for p in self._current_ai_polygon_points:
                    centroid += p
                centroid /= max(1, len(self._current_ai_polygon_points))
                self._ai_area_text_item.setPos(centroid + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))

    def set_ai_overlay_area(self, area_m2: float) -> None:
        """
        Update or create an area text overlay showing the area in square meters.
        """
        if area_m2 is None:
            # clear
            if self._ai_area_text_item:
                try:
                    self.scene.removeItem(self._ai_area_text_item)
                except Exception:
                    pass
                self._ai_area_text_item = None
            return
        text = f"{area_m2:.1f} m²"
        if self._ai_area_text_item is None:
            self._ai_area_text_item = QGraphicsSimpleTextItem(text)
            self._ai_area_text_item.setBrush(QBrush(QColor(255, 255, 255)))
            self._ai_area_text_item.setZValue(4)
            self.scene.addItem(self._ai_area_text_item)
        else:
            self._ai_area_text_item.setText(text)
        # Position at centroid if polygon exists
        if self._current_ai_polygon_points:
            centroid = QPointF(0.0, 0.0)
            for p in self._current_ai_polygon_points:
                centroid += p
            centroid /= max(1, len(self._current_ai_polygon_points))
            self._ai_area_text_item.setPos(centroid + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))

    def _flush_debounced_moves(self) -> None:
        """Emit any pending vertex moves after debounce interval."""
        try:
            pending = list(self._debounce_pending.items())
            self._debounce_pending.clear()
            for idx, pos in pending:
                self.ai_overlay_vertex_moved_debounced.emit(int(idx), pos)
        except Exception:
            pass

    def _select_ai_plane(self, index: int) -> None:
        """Select a detected AI plane for editing. Shows vertices for the selected plane and hides others."""
        if index == self._selected_ai_plane_index:
            return
        # Clear vertex visuals for previous selection
        for v in list(self._ai_vertex_items):
            try:
                self.scene.removeItem(v)
            except Exception:
                pass
        self._ai_vertex_items.clear()
        for l in list(self._ai_vertex_label_items):
            try:
                self.scene.removeItem(l)
            except Exception:
                pass
        self._ai_vertex_label_items.clear()
        # Reset polygon item visual for previous
        if 0 <= self._selected_ai_plane_index < len(self._ai_planes):
            prev = self._ai_planes[self._selected_ai_plane_index]
            try:
                prev['polygon_item'].setPen(QPen(prev.get('color', QColor(0,170,0)), 2))
            except Exception:
                pass
        # Update selection
        self._selected_ai_plane_index = index
        if not (0 <= index < len(self._ai_planes)):
            self._ai_polygon_item = None
            self._current_ai_polygon_points = []
            return
        plane = self._ai_planes[index]
        self._ai_polygon_item = plane['polygon_item']
        self._current_ai_polygon_points = list(plane['polygon_points'])
        # Highlight selected polygon
        try:
            highlight_pen = QPen(QColor(255, 255, 0), 3)
            self._ai_polygon_item.setPen(highlight_pen)
        except Exception:
            pass
        # Create vertex items and labels for selected polygon
        point_size = 8 / max(0.1, self._zoom_factor)
        for i, qp in enumerate(self._current_ai_polygon_points):
            v_item = DraggableVertexItem(i, qp, point_size, self)
            v_item.setZValue(2)
            self.scene.addItem(v_item)
            self._ai_vertex_items.append(v_item)

            label = QGraphicsSimpleTextItem(str(i+1))
            label.setBrush(QBrush(QColor(255, 255, 255)))
            label.setZValue(3)
            label.setPos(qp + QPointF(6/ self._zoom_factor, -12/ self._zoom_factor))
            self.scene.addItem(label)
            self._ai_vertex_label_items.append(label)

        for v in self._ai_vertex_items:
            try:
               v.enable_move_emits()
            except Exception:
               pass

        # Ensure area text exists
        if self._ai_area_text_item is None:
            area_text = QGraphicsSimpleTextItem("")
            area_text.setBrush(QBrush(QColor(255, 255, 255)))
            area_text.setZValue(4)
            self.scene.addItem(area_text)
            self._ai_area_text_item = area_text
        # Position area text
        centroid = QPointF(0.0, 0.0)
        for p in self._current_ai_polygon_points:
            centroid += p
        centroid /= max(1, len(self._current_ai_polygon_points))
        if self._ai_area_text_item:
            self._ai_area_text_item.setPos(centroid + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))

        # Emit selection for external handlers (as list of tuples)
        try:
            pts = [(float(p.x()), float(p.y())) for p in self._current_ai_polygon_points]
            self.ai_overlay_plane_selected.emit(index, pts)
        except Exception:
            pass

    def clear_ai_overlay_visuals(self) -> None:
        """
        Removes all AI overlay items from the scene.
        """
        for item in self._ai_overlay_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self._ai_overlay_items.clear()
        # Remove polygon and vertex items if present
        # Remove stored plane polygons
        for p in self._ai_planes:
            try:
                if 'polygon_item' in p and p['polygon_item']:
                    self.scene.removeItem(p['polygon_item'])
            except Exception:
                pass
        self._ai_planes.clear()
        if self._ai_polygon_item:
            try:
                self.scene.removeItem(self._ai_polygon_item)
            except Exception:
                pass
            self._ai_polygon_item = None
        for v in self._ai_vertex_items:
            try:
                self.scene.removeItem(v)
            except Exception:
                pass
        self._ai_vertex_items.clear()
        for l in self._ai_vertex_label_items:
            try:
                self.scene.removeItem(l)
            except Exception:
                pass
        self._ai_vertex_label_items.clear()
        self._current_ai_polygon_points.clear()
        # Remove area text if present
        if self._ai_area_text_item:
            try:
                self.scene.removeItem(self._ai_area_text_item)
            except Exception:
                pass
            self._ai_area_text_item = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handles mouse wheel events for zooming.
        """
        if not self._pixmap_item:
            super().wheelEvent(event)
            return

        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        old_pos = self.mapToScene(event.position().toPoint())

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
            self._zoom_factor *= zoom_in_factor
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
            self._zoom_factor *= zoom_out_factor

        new_pos = self.mapToScene(event.position().toPoint())

        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

        self.zoom_level_changed.emit(self._zoom_factor)
        event.accept()

        self.update_drawing_visuals(self._get_current_drawing_points_from_items())
        self.update_calibration_visuals() # Update calibration visuals on zoom
        # Update AI overlay visuals (vertex size and labels)
        try:
            self.update_ai_overlay_visuals()
        except Exception:
            pass

    def _get_current_drawing_points_from_items(self) -> List[Point2D]:
        """Helper to get Point2D list from current drawing items."""
        return [Point2D(item.rect().center().x(), item.rect().center().y()) for item in self._current_drawing_points_items]

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse press events for panning, drawing or calibration.
        """
        scene_pos = self.mapToScene(event.position().toPoint())
        logger.debug(f"mousePressEvent: calibration_mode_active={self._calibration_mode_active}, scene_pos={scene_pos}") # Diagnostic log
        self.mouse_pressed.emit(scene_pos, event.button())

        if not self._pixmap_item: # No image, no interaction
            super().mousePressEvent(event)
            return

        # If AI overlay is active and user clicked on an overlay polygon, select the plane
        if self._ai_overlay_active and not self._calibration_mode_active and not self._drawing_mode_active:
            items_at_pos = self.scene.items(scene_pos)
            for it in items_at_pos:
                # Find matching plane polygon
                for idx, plane in enumerate(self._ai_planes):
                    if it is plane.get('polygon_item'):
                        # Select this plane for editing
                        self._select_ai_plane(idx)
                        return

        if self._calibration_mode_active:
            if event.button() == Qt.MouseButton.LeftButton:
                self._calibration_points.append(scene_pos)
                logger.debug(f"Calibration mode: Point {len(self._calibration_points)} selected at {scene_pos}") # Diagnostic log
                self.update_calibration_visuals()
                if len(self._calibration_points) == 2:
                    self.calibration_points_selected.emit(
                        Point2D(self._calibration_points[0].x(), self._calibration_points[0].y()),
                        Point2D(self._calibration_points[1].x(), self._calibration_points[1].y())
                    )
                    logger.debug("Calibration mode: calibration_points_selected.emit called.") # Diagnostic log
                    self.set_calibration_mode(False) # Exit calibration mode after selecting points
            return # Consume event in calibration mode

        if self._drawing_mode_active:
            if event.button() == Qt.MouseButton.LeftButton:
                self._selected_point_item = None
                self._selected_point_index = -1
                for i, item in enumerate(self._current_drawing_points_items):
                    if item.contains(scene_pos):
                        self._selected_point_item = item
                        self._selected_point_index = i
                        self._is_dragging_point = True
                        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                        self.update_drawing_visuals(self._get_current_drawing_points_from_items())
                        return
                
                self.point_added_to_drawing.emit(scene_pos)
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
                self._is_dragging_point = False
            elif event.button() == Qt.MouseButton.RightButton:
                self.polygon_drawing_finished.emit()
                self.set_drawing_mode(False)
            return # Consume event in drawing mode

        # Default panning behavior if no other mode is active
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._last_pan_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handles mouse move events for panning, drawing or calibration.
        """
        scene_pos = self.mapToScene(event.position().toPoint())
        self.mouse_moved.emit(scene_pos)

        if not self._pixmap_item: # No image, no interaction
            super().mouseMoveEvent(event)
            return

        if self._calibration_mode_active and len(self._calibration_points) == 1:
            # Draw rubber band line from first point to current mouse position
            p1 = self._calibration_points[0]
            if not self._calibration_line_item:
                self._calibration_line_item = QGraphicsLineItem(p1.x(), p1.y(), scene_pos.x(), scene_pos.y())
                self._calibration_line_item.setPen(self._calibration_line_pen)
                self.scene.addItem(self._calibration_line_item)
            else:
                self._calibration_line_item.setLine(p1.x(), p1.y(), scene_pos.x(), scene_pos.y())
            return # Consume event in calibration mode

        if self._drawing_mode_active:
            if self._is_dragging_point and self._selected_point_index != -1:
                self.point_moved_in_drawing.emit(self._selected_point_index, scene_pos)
            elif len(self._current_drawing_points_items) > 0 and not self._is_dragging_point:
                last_point_pos = self._current_drawing_points_items[-1].rect().center()
                if not self._rubber_band_line:
                    self._rubber_band_line = QGraphicsLineItem(last_point_pos.x(), last_point_pos.y(), scene_pos.x(), scene_pos.y())
                    self._rubber_band_line.setPen(self._rubber_band_pen)
                    self.scene.addItem(self._rubber_band_line)
                else:
                    self._rubber_band_line.setLine(last_point_pos.x(), last_point_pos.y(), scene_pos.x(), scene_pos.y())
            return # Consume event in drawing mode

        # Default panning behavior
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

        if not self._pixmap_item: # No image, no interaction
            super().mouseReleaseEvent(event)
            return

        if self._calibration_mode_active:
            # No specific action on mouse release in calibration mode, just consume event
            return

        if self._drawing_mode_active:
            if event.button() == Qt.MouseButton.LeftButton:
                self._is_dragging_point = False
                self._selected_point_item = None
                self._selected_point_index = -1
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
                self.update_drawing_visuals(self._get_current_drawing_points_from_items())
            return # Consume event in drawing mode

        # Default panning behavior
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