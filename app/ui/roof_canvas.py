"""
This module defines the RoofCanvas widget, a QGraphicsView for displaying roof images
and enabling interactive geometry editing.
"""

from typing import Optional, List, Union
import numpy as np
import cv2 # For converting mask to QImage
import math

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QWidget, QSizePolicy,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPolygonItem, QGraphicsSimpleTextItem, QGraphicsItem,
    QGraphicsRectItem
)
from PySide6.QtGui import QPixmap, QTransform, QMouseEvent, QWheelEvent, QPen, QBrush, QColor, QImage, QPolygonF, QPainter
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QTimer
from app.core.logger import setup_logging # Import logger

from app.core.image.image_model import ImageInfo
from app.geometry.point import Point2D
from app.geometry.polygon import Polygon2D
from app.ai.segmentation_result import SegmentationResult
from app.ai.ai_result import DetectionResult, BoundingBox, PolygonGeometry

logger = setup_logging() # Initialize logger

# ---- Pomocne funkcie pre klasifikaciu narozie/uzlabie (Claude fix) ----

def _polygon_signed_area(pts):
    """Shoelace: positive = CCW, negative = CW."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0

def _corner_turn(perimeter, idx):
    """Cross product zákruty pri vrchole perimeter[idx]."""
    n = len(perimeter)
    prev_p = perimeter[(idx - 1) % n]
    cur_p = perimeter[idx]
    next_p = perimeter[(idx + 1) % n]
    v1 = (cur_p[0] - prev_p[0], cur_p[1] - prev_p[1])
    v2 = (next_p[0] - cur_p[0], next_p[1] - cur_p[1])
    return v1[0] * v2[1] - v1[1] * v2[0]

def _find_nearest_perimeter_index(point, perimeter, tolerance=3.0):
    """Nájde index vrcholu v perimeter najbližšie k point (v rámci tolerancie px)."""
    best_idx, best_dist = None, tolerance
    for i, p in enumerate(perimeter):
        d = math.hypot(point[0] - p[0], point[1] - p[1])
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx

def _classify_hip_or_valley(p1, p2, perimeter, tolerance=3.0):
    """Robustná klasifikacia narozie/uzlabie podla vonkajsieho obrysu."""
    if len(perimeter) < 3:
        return "narozie"
    area = _polygon_signed_area(perimeter)
    if abs(area) < 1e-6:
        return "narozie"
    ccw_sign = 1.0 if area > 0 else -1.0
    for pt in (p1, p2):
        idx = _find_nearest_perimeter_index(pt, perimeter, tolerance)
        if idx is not None:
            turn = _corner_turn(perimeter, idx)
            is_convex = (turn * ccw_sign) > 0
            return "narozie" if is_convex else "uzlabie"
    for pt in (p1, p2):
        idx = _find_nearest_perimeter_index(pt, perimeter, tolerance * 3)
        if idx is not None:
            turn = _corner_turn(perimeter, idx)
            is_convex = (turn * ccw_sign) > 0
            return "narozie" if is_convex else "uzlabie"
    return "narozie"

def _merge_close_vertices(boundary_edges, merge_tol=4.0):
    """Zlucenie blizkych vrcholov union-find algoritmom pred retazenim hranic."""
    points = []
    for p1, p2 in boundary_edges:
        points.append((p1.x(), p1.y()))
        points.append((p2.x(), p2.y()))
    n = len(points)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if (dx * dx + dy * dy) ** 0.5 <= merge_tol:
                union(i, j)
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(points[i])
    representative = {}
    for r, pts in groups.items():
        avg_x = sum(p[0] for p in pts) / len(pts)
        avg_y = sum(p[1] for p in pts) / len(pts)
        representative[r] = QPointF(avg_x, avg_y)
    merged_edges = []
    idx = 0
    for p1, p2 in boundary_edges:
        r1 = find(idx); idx += 1
        r2 = find(idx); idx += 1
        merged_edges.append((representative[r1], representative[r2]))
    return merged_edges


def _build_true_outer_perimeter(edge_planes):
    '''Posklada skutocny (aj reflexny) obrys celej strechy z hran s 1 vlastnikom.'''
    boundary_edges = []
    for key, owners in edge_planes.items():
        if len(owners) == 1:
            _, p1, p2 = owners[0]
            boundary_edges.append((p1, p2))
    if len(boundary_edges) < 3:
        return []

    # OPRAVA: zluc takmer-identicke vrcholy z roznych planov PRED retazenim
    # hranic, inak drobne AI nepresnosti roztrhnu obrys a vyzaduje si to
    # rucne mazanie bodov.
    boundary_edges = _merge_close_vertices(boundary_edges, merge_tol=4.0)

    def key_of(pt):
        return (round(pt.x(), 2), round(pt.y(), 2))
    adjacency = {}
    for p1, p2 in boundary_edges:
        k1, k2 = key_of(p1), key_of(p2)
        adjacency.setdefault(k1, []).append((k2, p2))
        adjacency.setdefault(k2, []).append((k1, p1))
    bad_vertices = [k for k, v in adjacency.items() if len(v) != 2]
    if bad_vertices:
        import logging
        logging.getLogger("Roof AI Studio").warning(
            "_build_true_outer_perimeter: %d vrcholov nema presne 2 hranicne hrany - %s",
            len(bad_vertices), bad_vertices[:5])
    if not adjacency:
        return []
    start_key = next(iter(adjacency))
    start_pt = None
    for p1, p2 in boundary_edges:
        if key_of(p1) == start_key:
            start_pt = p1
            break
    if start_pt is None:
        return []
    ring = [start_pt]
    visited = {start_key}
    prev_key, cur_key = None, start_key
    for _ in range(len(adjacency) + 2):
        neighbors = adjacency.get(cur_key, [])
        nxt = None
        for nk, npt in neighbors:
            if nk != prev_key:
                nxt = (nk, npt)
                break
        if nxt is None:
            break
        nk, npt = nxt
        if nk == start_key:
            break
        if nk in visited:
            break
        ring.append(npt)
        visited.add(nk)
        prev_key, cur_key = cur_key, nk
    return ring

def _classify_hip_or_valley_fallback(p1, p2, owners_data, planes_data):
    """Fallback: segmentový test bez perimeter dát."""
    TOL = 3.0
    def find_far_vertex(plane_idx, ax, ay, bx, by):
        pts = planes_data[plane_idx]["polygon_points"]
        mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
        best_pt, best_d = None, -1.0
        for pt in pts:
            px, py = pt.x(), pt.y()
            near_a = math.hypot(px - ax, py - ay) < TOL
            near_b = math.hypot(px - bx, py - by) < TOL
            if near_a or near_b:
                continue
            d = math.hypot(px - mx, py - my)
            if d > best_d:
                best_d, best_pt = d, (px, py)
        return best_pt

    def ccw(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def segments_intersect(p, q, r, s):
        d1 = ccw(r, s, p); d2 = ccw(r, s, q)
        d3 = ccw(p, q, r); d4 = ccw(p, q, s)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    ax, ay, bx, by = p1.x(), p1.y(), p2.x(), p2.y()
    idx_a = owners_data[0][0]; idx_b = owners_data[1][0]
    far_a = find_far_vertex(idx_a, ax, ay, bx, by)
    far_b = find_far_vertex(idx_b, ax, ay, bx, by)
    if far_a is None or far_b is None:
        return "narozie"
    if segments_intersect(far_a, far_b, (ax, ay), (bx, by)):
        return "uzlabie"
    return "narozie"

# ---- Koniec pomocnych funkcii ----



class DraggableVertexItem(QGraphicsEllipseItem):
    """
    Small draggable vertex item used for AI overlay. Stores an index and notifies
    parent RoofCanvas on position changes via right-click drag.
    """
    def __init__(self, index: int, pos: QPointF, size: float, canvas: 'RoofCanvas'):
        super().__init__(-size/2, -size/2, size, size)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.RightButton)
        self.setAcceptHoverEvents(True)
        self._index = index
        self._canvas = canvas
        self._dragging = False
        self._drag_start = QPointF()
        self._drag_start_pos = QPointF()
        self.setPen(canvas._point_pen)
        self.setBrush(canvas._point_brush)
        self._suppress_move_emit = True
        self.setPos(pos)

    def enable_move_emits(self) -> None:
        self._suppress_move_emit = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._dragging = True
            self._drag_start = event.scenePos()
            self._drag_start_pos = self.scenePos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and (event.buttons() & Qt.MouseButton.RightButton):
            delta = event.scenePos() - self._drag_start
            new_pos = self._drag_start_pos + delta
            self.setPos(new_pos)
            if not self._suppress_move_emit:
                self._canvas.ai_overlay_vertex_moved.emit(self._index, new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._dragging:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.CursorShape.CrossCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)


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
    polygon_drawn_for_measurement = Signal(list) # New signal for manual polygon measurement

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
        self._ai_overlay_px_per_m: float = 1.0  # scale for area recalc
        self._outer_perimeter_items: list = []
        self._show_outer_perimeter: bool = True
        self._building_centroid = None
        self._outer_perimeter_points: list = []
        self._ai_overlay_items: List[Union[QGraphicsPolygonItem, QGraphicsPixmapItem, QGraphicsRectItem]] = []
        # Support multiple detected planes: list of dicts {polygon_item, polygon_points, color}
        self._ai_planes: List[dict] = []
        self._selected_ai_plane_index: int = -1
        # Polygon overlay and draggable vertices for the SELECTED plane
        self._ai_polygon_item: Optional[QGraphicsPolygonItem] = None
        self._ai_vertex_items: List[DraggableVertexItem] = []
        self._ai_vertex_label_items: List[QGraphicsSimpleTextItem] = []
        self._ai_edge_label_items: List[QGraphicsSimpleTextItem] = []
        self._current_ai_polygon_points: List[QPointF] = []
        # Area text item for AI overlay (shared, shows area for selected plane)
        self._ai_area_text_item: Optional[QGraphicsSimpleTextItem] = None
        # Debounce infrastructure for vertex moved events
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_interval_ms = 100
        self._debounce_timer.timeout.connect(self._flush_debounced_moves)
        self._debounce_pending: dict = {}
        self._edge_class_overrides: dict = {}  # edge_key -> label_sk
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

    def set_ai_overlay_scale(self, px_per_m: float) -> None:
        """Set the pixel-per-meter scale for area/perimeter recalculation."""
        self._ai_overlay_px_per_m = px_per_m

    def set_ai_overlay_area(self, area_m2: float) -> None:
        """Update AI overlay area display text."""
        pass  # no-op for now, area shown via other UI elements

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

    def _recalc_dimensions(self, pts_px: list) -> None:
        """Recalculate area and perimeter from polygon points using stored scale."""
        if len(pts_px) < 3:
            return
        # Shoelace formula for area in pixels
        area_px = 0.0
        n = len(pts_px)
        for i in range(n):
            x1, y1 = pts_px[i]
            x2, y2 = pts_px[(i + 1) % n]
            area_px += x1 * y2 - x2 * y1
        area_px = abs(area_px) / 2.0
        # Perimeter in pixels
        peri_px = 0.0
        for i in range(n):
            x1, y1 = pts_px[i]
            x2, y2 = pts_px[(i + 1) % n]
            peri_px += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        # Convert to meters
        s = self._ai_overlay_px_per_m
        area_m2 = area_px / (s * s)
        peri_m = peri_px / s
        # Edge lengths
        edge_lens = []
        for i in range(n):
            x1, y1 = pts_px[i]
            x2, y2 = pts_px[(i + 1) % n]
            edge_lens.append(round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / s, 1))

        # Update area text
        if self._ai_area_text_item is None:
            area_text = QGraphicsSimpleTextItem("")
            area_text.setBrush(QBrush(QColor(255, 255, 255)))
            area_text.setZValue(4)
            self.scene.addItem(area_text)
            self._ai_area_text_item = area_text

        centroid_x = sum(p[0] for p in pts_px) / n
        centroid_y = sum(p[1] for p in pts_px) / n
        text = f"{area_m2:.1f}m2  P={peri_m:.1f}m  hrany: {edge_lens}"
        self._ai_area_text_item.setText(text)
        self._ai_area_text_item.setPos(
            centroid_x + 6 / max(0.1, self._zoom_factor),
            centroid_y - 24 / max(0.1, self._zoom_factor)
        )

    def _on_ai_overlay_vertex_moved(self, index: int, scene_pos: QPointF) -> None:
        """
        Internal handler when a vertex is moved by user. Update polygon visual immediately.
        Also schedule a debounced emission for external consumers.
        """
        if not (0 <= index < len(self._current_ai_polygon_points)):
            return
        # Update point immediately for visual feedback
        self._current_ai_polygon_points[index] = scene_pos
        # --- SNAP LOGIC ---
        snap = self._find_snap_target(scene_pos, self._selected_ai_plane_index, index)
        if snap:
            scene_pos = snap["position"]
            if snap["same_plane"]:
                del_idx = snap["vertex_idx"]
                if del_idx != index and len(self._current_ai_polygon_points) > 3:
                    self._current_ai_polygon_points[index] = scene_pos
                    drag_idx = index - 1 if del_idx < index else index
                    del self._current_ai_polygon_points[del_idx]
                    self._ai_polygon_item.setPolygon(QPolygonF(self._current_ai_polygon_points))
                    if del_idx < len(self._ai_vertex_items):
                        try: self.scene.removeItem(self._ai_vertex_items[del_idx])
                        except: pass
                        del self._ai_vertex_items[del_idx]
                    if del_idx < len(self._ai_vertex_label_items):
                        try: self.scene.removeItem(self._ai_vertex_label_items[del_idx])
                        except: pass
                        del self._ai_vertex_label_items[del_idx]
                    for vi, vitem in enumerate(self._ai_vertex_items):
                        vitem._index = vi
                    if drag_idx < len(self._ai_vertex_items):
                        self._ai_vertex_items[drag_idx].setPos(scene_pos)
                        self._ai_vertex_items[drag_idx]._dragging = True
                    if drag_idx < len(self._ai_vertex_label_items):
                        self._ai_vertex_label_items[drag_idx].setPos(
                            scene_pos + QPointF(6.0/max(0.1,self._zoom_factor), -12.0/max(0.1,self._zoom_factor)))
                    pts = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
                    self._recalc_dimensions(pts)
                    self._sync_edited_polygon_to_planes()
                    self._mark_snapped_vertices(self._selected_ai_plane_index, drag_idx)
                    return
                else:
                    self._current_ai_polygon_points[index] = scene_pos
                    self._ai_polygon_item.setPolygon(QPolygonF(self._current_ai_polygon_points))
                    pts = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
                    self._recalc_dimensions(pts)
                    self._sync_edited_polygon_to_planes()
            else:
                self._current_ai_polygon_points[index] = scene_pos
                self._ai_polygon_item.setPolygon(QPolygonF(self._current_ai_polygon_points))
                pts = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
                self._recalc_dimensions(pts)
                self._sync_edited_polygon_to_planes()
                self._update_plane_polygon(snap["plane_idx"])
                self._mark_snapped_vertices(snap["plane_idx"], snap["vertex_idx"])
        else:
            if self._ai_polygon_item:
                try:
                    self._ai_polygon_item.setPolygon(QPolygonF(self._current_ai_polygon_points))
                    pts = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
                    self._recalc_dimensions(pts)
                    self._sync_edited_polygon_to_planes()
                except Exception:
                    pass
        # Update label position
        if index < len(self._ai_vertex_label_items):
            label = self._ai_vertex_label_items[index]
            label.setPos(scene_pos + QPointF(6.0 / max(0.1, self._zoom_factor), -12.0 / max(0.1, self._zoom_factor)))

        # Update edge labels
        self._update_edge_labels()

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
                polygon_pts = None
                
                # Check for structured geometry
                if hasattr(result, "geometry") and isinstance(result.geometry, PolygonGeometry):
                    polygon_pts = result.geometry.vertices
                
                # Fallback to metadata
                if polygon_pts is None and isinstance(result.metadata, dict):
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

                    # Do NOT auto-select; user clicks to select

                    plane_index += 1

                else:
                    rect = QRectF(bbox.x_min, bbox.y_min, bbox.width, bbox.height)
                    rect_item = self.scene.addRect(rect, self._detection_box_pen)
                    self._ai_overlay_items.append(rect_item)


        self._update_outer_perimeter()
        for item in self._ai_overlay_items:
            item.setZValue(1)
        # Ensure vertex items and labels have proper Z
        for v in self._ai_vertex_items:
            v.setZValue(2)
        for l in self._ai_vertex_label_items:
            l.setZValue(3)

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
        for el in self._ai_edge_label_items:
            try: self.scene.removeItem(el)
            except: pass
        self._ai_edge_label_items.clear()
        self._current_ai_polygon_points.clear()
        # Remove area text if present
        if self._ai_area_text_item:
            try:
                self.scene.removeItem(self._ai_area_text_item)
            except Exception:
                pass
            self._ai_area_text_item = None

    def _flush_debounced_moves(self) -> None:
        """Emit any pending vertex moves after debounce interval."""
        try:
            pending = list(self._debounce_pending.items())
            self._debounce_pending.clear()
            for idx, pos in pending:
                self.ai_overlay_vertex_moved_debounced.emit(int(idx), pos)
        except Exception:
            pass

    def _find_snap_target(self, pos, exclude_plane_idx, exclude_vertex_idx, threshold_px=11.0):
        """Find nearest vertex across ALL planes for snapping."""
        from PySide6.QtCore import QPointF
        drag_origin = None
        if 0 <= exclude_plane_idx < len(self._ai_planes):
            opts = self._ai_planes[exclude_plane_idx].get("polygon_points", [])
            if exclude_vertex_idx < len(opts):
                drag_origin = (opts[exclude_vertex_idx].x(), opts[exclude_vertex_idx].y())
        best_dist = threshold_px
        best = None
        for pi, plane in enumerate(self._ai_planes):
            pts = plane.get("polygon_points", [])
            for vi, pt in enumerate(pts):
                if pi == exclude_plane_idx and vi == exclude_vertex_idx:
                    continue
                if drag_origin is not None:
                    if abs(pt.x() - drag_origin[0]) < 0.5 and abs(pt.y() - drag_origin[1]) < 0.5:
                        continue
                d = ((pt.x() - pos.x())**2 + (pt.y() - pos.y())**2)**0.5
                if d < best_dist:
                    best_dist = d
                    best = {"plane_idx": pi, "vertex_idx": vi, "position": QPointF(pt.x(), pt.y()),
                            "same_plane": pi == exclude_plane_idx, "distance": d}
        return best

    def _update_plane_polygon(self, plane_idx):
        if not (0 <= plane_idx < len(self._ai_planes)):
            return
        from PySide6.QtGui import QPolygonF
        plane = self._ai_planes[plane_idx]
        plane["polygon_item"].setPolygon(QPolygonF(plane["polygon_points"]))
        if plane_idx == self._selected_ai_plane_index:
            pts = [(p.x(), p.y()) for p in plane["polygon_points"]]
            self._recalc_dimensions(pts)

    def _mark_snapped_vertices(self, plane_idx, vertex_idx):
        if not (0 <= plane_idx < len(self._ai_planes)):
            return
        pts = self._ai_planes[plane_idx].get("polygon_points", [])
        if vertex_idx >= len(pts):
            return
        from PySide6.QtCore import QTimer
        from PySide6.QtGui import QPen, QBrush, QColor
        sp = pts[vertex_idx]
        ring = QGraphicsEllipseItem(sp.x()-6, sp.y()-6, 12, 12)
        ring.setPen(QPen(QColor(0,255,0,200), 2))
        ring.setBrush(QBrush(QColor(0,255,0,60)))
        ring.setZValue(5)
        self.scene.addItem(ring)
        def _rm(): 
            try: self.scene.removeItem(ring)
            except: pass
        QTimer.singleShot(400, _rm)

    def _sync_edited_polygon_to_planes(self) -> None:
        """Persist current polygon points back to _ai_planes source data."""
        if self._selected_ai_plane_index < 0 or not self._ai_planes:
            return
        stored = self._ai_planes[self._selected_ai_plane_index]
        stored["polygon_points"] = list(self._current_ai_polygon_points)
        self._update_outer_perimeter()

    def _compute_convex_hull_edges(self, all_pts):
        if len(all_pts) < 3:
            return set()
        pts_sorted = sorted(all_pts)
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        hull_edges = set()
        n = len(hull)
        for i in range(n):
            p1, p2 = hull[i], hull[(i+1)%n]
            if p1 < p2:
                hull_edges.add((p1[0], p1[1], p2[0], p2[1]))
            else:
                hull_edges.add((p2[0], p2[1], p1[0], p1[1]))
        return hull_edges

    def _compute_dominant_okap_direction(self, edge_planes, hull_edges_set):
        import math
        angles = []
        for key, owners in edge_planes.items():
            if len(owners) != 1 or key not in hull_edges_set:
                continue
            x1, y1, x2, y2 = key
            dx, dy = x2 - x1, y2 - y1
            length = (dx*dx + dy*dy)**0.5
            if length < 5:
                continue
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += math.pi
            angles.append((angle, length))
        if not angles:
            return None
        bins = 18
        hist = [0.0] * bins
        for angle, weight in angles:
            hist[int(angle / math.pi * bins) % bins] += weight
        return (max(range(bins), key=lambda i: hist[i]) + 0.5) * math.pi / bins

    def _classify_edge(self, p1, p2, centroids, hull_edges_set, key, okap_dir=None, owners_data=None, planes_data=None):
        """Klasifikacia hrany. Vracia (label_sk, (r,g,b), width)."""
        import math
        ANGLE_TOL = math.radians(15)

        def edge_angle(a, b):
            ang = math.atan2(b.y() - a.y(), b.x() - a.x())
            return ang % math.pi

        def is_parallel_to_okap(ang):
            if okap_dir is None:
                return False
            diff = abs(ang - okap_dir)
            diff = min(diff, math.pi - diff)
            return diff <= ANGLE_TOL

        edge_ang = edge_angle(p1, p2)
        px1, py1 = p1.x(), p1.y()
        px2, py2 = p2.x(), p2.y()
        mx, my = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        ex, ey = px2 - px1, py2 - py1
        edge_len = (ex*ex + ey*ey)**0.5
        if edge_len < 0.01:
            return ("", (128,128,128), 1)
        nx, ny = -ey, ey  # using ex for ny? No: nx=-dy, ny=dx
        nx, ny = -ey / edge_len, ex / edge_len

        # ---- 1 polygon: outer edge ----
        if len(centroids) == 1:
            cx, cy = centroids[0]
            if self._building_centroid is not None:
                bcx, bcy = self._building_centroid
                build_side = (bcx - mx) * nx + (bcy - my) * ny
                if build_side > 0:
                    nx, ny = -nx, -ny
                plane_side = (cx - mx) * nx + (cy - my) * ny
                if plane_side < 0:
                    return ("okap", (0, 240, 255), 4)
                else:
                    return ("stit", (100, 255, 130), 2)
            return ("okap", (0, 240, 255), 4) if edge_len > 30 else ("stit", (100, 255, 130), 2)

        # ---- 2 polygons: shared edge ----
        if len(centroids) != 2:
            return ("internal", (255, 200, 50), 1.5)

        # OPRAVA v2: predosly test cez znamienko taziska (same_side = s1*s2>0)
        # bol nespolahlivy - pri nesymetrickych AI-detegovanych polygonoch sa
        # lahko "prehodi" na nespravnu stranu aj pre skutocny hreben, co viedlo
        # raz k falosnemu hrebenu (T-junction hrana), inokedy k uplnemu
        # zmiznutiu skutocneho hrebena.
        #
        # Namiesto toho pouzi vlastnost, ktora je geometricky stabilna:
        # SKUTOCNY HREBEN lezi striktne VNUTRI strechy - ani jeden jeho
        # koncovy bod sa nedotyka vonkajsieho obrysu (na rozdiel od narozi/
        # uzlabia, ktore vzdy maju aspon jeden koniec na okape/rohu podorysu).
        perimeter = getattr(self, "_outer_perimeter_points", None)
        perimeter_xy = [(pt.x(), pt.y()) for pt in perimeter] if perimeter else []

        def _on_perimeter(pt_xy):
            if not perimeter_xy:
                return False
            return _find_nearest_perimeter_index(pt_xy, perimeter_xy, tolerance=3.0) is not None

        p1_on_perim = _on_perimeter((px1, py1))
        p2_on_perim = _on_perimeter((px2, py2))

        if (not p1_on_perim) and (not p2_on_perim) and is_parallel_to_okap(edge_ang):
            return ("hreben", (60, 60, 60), 3.5)

        if p1_on_perim or p2_on_perim:
            # aspon jeden koniec sa dotyka obrysu -> narozie/uzlabie
            if perimeter_xy:
                result = _classify_hip_or_valley((px1, py1), (px2, py2), perimeter_xy, tolerance=3.0)
            else:
                result = _classify_hip_or_valley_fallback(p1, p2, owners_data, planes_data)
            if result == "narozie":
                return ("narozie", (255, 80, 60), 2.5)
            else:
                return ("uzlabie", (180, 80, 255), 2)

        # Ani jeden koniec nie je na obryse, ale hrana NIE JE rovnobezna
        # s okapom - vzacny pripad (napr. vnutorny "T-spoj" dvoch hrebenov
        # roznej vysky). Zarad ako hreben (je to stale vnutorna "vysoka"
        # hrana), ale zaloguj to pre kontrolu.
        logger.debug(
            "_classify_edge: interny T-spoj hrebenov (ani jeden koniec na "
            "perimetri, uhol nie je rovnobezny s okapom) p1=(%.1f,%.1f) "
            "p2=(%.1f,%.1f) edge_ang=%.1fdeg okap_dir=%s",
            px1, py1, px2, py2, math.degrees(edge_ang),
            f"{math.degrees(okap_dir):.1f}deg" if okap_dir is not None else "None"
        )
        return ("hreben", (60, 60, 60), 3.5)

    def _update_outer_perimeter(self) -> None:
        """Classify edges: okap(cyan), stit(green), hreben(dark/org), uzlabie(purple), narozie(red)."""
        for item in self._outer_perimeter_items:
            try: self.scene.removeItem(item)
            except: pass
        self._outer_perimeter_items.clear()
        import logging
        logger = logging.getLogger("Roof AI Studio")
        if not self._show_outer_perimeter or not self._ai_planes:
            logger.debug("_update_outer_perimeter: skipped (show=%s, planes=%d)", self._show_outer_perimeter, len(self._ai_planes))
            return
        from collections import defaultdict
        from PySide6.QtGui import QPen, QColor, QBrush
        edge_planes = defaultdict(list)
        plane_centroids = {}
        for pi, plane in enumerate(self._ai_planes):
            pts = plane.get("polygon_points", [])
            n_pts = len(pts)
            if n_pts < 3:
                continue
            cx = sum(p.x() for p in pts) / n_pts
            cy = sum(p.y() for p in pts) / n_pts
            plane_centroids[pi] = (cx, cy)
            for i in range(n_pts):
                j = (i + 1) % n_pts
                p1, p2 = pts[i], pts[j]
                if (p1.x() < p2.x()) or (p1.x() == p2.x() and p1.y() < p2.y()):
                    key = (p1.x(), p1.y(), p2.x(), p2.y())
                else:
                    key = (p2.x(), p2.y(), p1.x(), p1.y())
                edge_planes[key].append((pi, p1, p2))
        all_pts = []
        for plane in self._ai_planes:
            for pt in plane.get("polygon_points", []):
                all_pts.append((pt.x(), pt.y()))
        hull = self._compute_convex_hull_edges(all_pts)
        hull_edges_set = hull
        # Also store the ordered perimeter points for narozie/uzlabie classification
        if all_pts:
            from collections import OrderedDict
            pts_sorted = sorted(all_pts)
            def _cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            lower = []
            for p in pts_sorted:
                while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
                    lower.pop()
                lower.append(p)
            upper = []
            for p in reversed(pts_sorted):
                while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
                    upper.pop()
                upper.append(p)
            hull_pts = lower[:-1] + upper[:-1]
            # Use TRUE outer ring (possibly reflex) instead of convex hull
            # Convex hull "zjeda" reflexne rohy -> uzlabie by sa nikdy nedetegovalo
            true_ring = _build_true_outer_perimeter(edge_planes)
            if true_ring:
                from PySide6.QtCore import QPointF as _QPF
                self._outer_perimeter_points = [_QPF(pt.x(), pt.y()) for pt in true_ring]
            else:
                from PySide6.QtCore import QPointF as _QPF
                self._outer_perimeter_points = [_QPF(x, y) for x, y in hull_pts]
        else:
            self._outer_perimeter_points = []
        if plane_centroids:
            all_bc = list(plane_centroids.values())
            self._building_centroid = (sum(c[0] for c in all_bc) / len(all_bc),
                                       sum(c[1] for c in all_bc) / len(all_bc))
        okap_dir = self._compute_dominant_okap_direction(edge_planes, hull_edges_set)
        label_counts = {}
        for key, owners in edge_planes.items():
            p1, p2 = owners[0][1], owners[0][2]
            centroids = [plane_centroids[pi] for pi, _, _ in owners]
            label_sk, (r,g,b), width = self._classify_edge(p1, p2, centroids, hull_edges_set, key, okap_dir, owners, self._ai_planes)
            # Apply user override if exists
            override = self._edge_class_overrides.get(key)
            if override:
                label_sk = override
                # Re-resolve color from override
                color_map = {"okap": (0,240,255), "stit": (100,255,130), "hreben": (60,60,60),
                             "narozie": (255,80,60), "uzlabie": (180,80,255), "internal": (255,200,50)}
                r, g, b = color_map.get(override, (128,128,128))
            if not label_sk:
                continue
            pen = QPen(QColor(r, g, b, 220), width)
            line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            line.setPen(pen)
            line.setZValue(3.5)
            self.scene.addItem(line)
            self._outer_perimeter_items.append(line)
            label_counts[label_sk] = (r, g, b)
        # Store classification for export (legend hidden from canvas)
        self._edge_classification = label_counts
        logger.info("_update_outer_perimeter: classified %d edges, %d types", len(self._outer_perimeter_items) - len(label_counts), len(label_counts))

    def toggle_outer_perimeter(self) -> None:
        self._show_outer_perimeter = not self._show_outer_perimeter
        self._update_outer_perimeter()


    def _delete_selected_polygon(self) -> None:
        """Delete the currently selected AI plane polygon."""
        idx = self._selected_ai_plane_index
        if idx < 0 or idx >= len(self._ai_planes):
            return
        plane = self._ai_planes[idx]
        try:
            if 'polygon_item' in plane and plane['polygon_item']:
                self.scene.removeItem(plane['polygon_item'])
        except Exception:
            pass
        for v in self._ai_vertex_items:
            try: self.scene.removeItem(v)
            except: pass
        self._ai_vertex_items.clear()
        for lb in self._ai_vertex_label_items:
            try: self.scene.removeItem(lb)
            except: pass
        self._ai_vertex_label_items.clear()
        if self._ai_area_text_item:
            try: self.scene.removeItem(self._ai_area_text_item)
            except: pass
            self._ai_area_text_item = None
        del self._ai_planes[idx]
        self._selected_ai_plane_index = -1
        self._ai_polygon_item = None
        self._current_ai_polygon_points = []
        self._update_outer_perimeter()


    # --- Add-plane mode ---
    def _toggle_add_plane_mode(self) -> None:
        """Toggle add-plane drawing mode. Draw a new polygon and add it to AI planes."""
        self._add_plane_mode = not getattr(self, '_add_plane_mode', False)
        if self._add_plane_mode:
            self._add_plane_points = []
            self._add_plane_items = []
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.status_bar_msg = "Add plane: left-click = add vertex, right-click = finish"
            print("Add-plane mode ON")
        else:
            self._clear_add_plane_visuals()
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            print("Add-plane mode OFF")

    def _clear_add_plane_visuals(self) -> None:
        for item in getattr(self, '_add_plane_items', []):
            try: self.scene.removeItem(item)
            except: pass
        self._add_plane_items = []
        self._add_plane_points = []

    def _finalize_add_plane(self) -> None:
        """Finish drawing and add new polygon to AI planes."""
        pts = self._add_plane_points
        self._clear_add_plane_visuals()
        self._add_plane_mode = False
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        if len(pts) < 3:
            return
        # Create polygon item
        qpoly = QPolygonF(pts)
        color = self._ai_plane_colors[len(self._ai_planes) % len(self._ai_plane_colors)]
        pen = QPen(color, 2)
        poly_item = QGraphicsPolygonItem(qpoly)
        poly_item.setPen(pen)
        poly_item.setBrush(Qt.NoBrush)
        poly_item.setZValue(1)
        self.scene.addItem(poly_item)
        self._ai_overlay_items.append(poly_item)
        self._ai_planes.append({
            'polygon_item': poly_item,
            'polygon_points': pts,
            'color': color,
        })
        self._update_outer_perimeter()
        # Select the new plane
        self._select_ai_plane(len(self._ai_planes) - 1)


    def _merge_polygons(self, idx_a: int, idx_b: int) -> None:
        """Merge two adjacent AI planes into one by computing convex hull."""
        if idx_a == idx_b:
            return
        a, b = min(idx_a, idx_b), max(idx_a, idx_b)
        plane_a = self._ai_planes[a]
        plane_b = self._ai_planes[b]
        pts_a = [(p.x(), p.y()) for p in plane_a['polygon_points']]
        pts_b = [(p.x(), p.y()) for p in plane_b['polygon_points']]
        # Convex hull of combined points
        all_pts = pts_a + pts_b
        if len(all_pts) < 3:
            return
        pts_sorted = sorted(set(all_pts))
        def cross(o, x, y):
            return (x[0]-o[0])*(y[1]-o[1]) - (x[1]-o[1])*(y[0]-o[0])
        lower = []
        for p in pts_sorted:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts_sorted):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        # Remove plane B first (higher index)
        self._delete_plane(b)
        if a > b:
            a -= 1
        # Replace plane A with merged hull
        plane_a['polygon_points'] = [QPointF(x, y) for x, y in hull]
        plane_a['polygon_item'].setPolygon(QPolygonF(plane_a['polygon_points']))
        self._update_outer_perimeter()
        self._select_ai_plane(a)

    def _delete_plane(self, idx: int) -> None:
        """Remove a plane by index without UI cleanup."""
        if idx < 0 or idx >= len(self._ai_planes):
            return
        plane = self._ai_planes[idx]
        try:
            if 'polygon_item' in plane and plane['polygon_item']:
                self.scene.removeItem(plane['polygon_item'])
        except: pass
        del self._ai_planes[idx]


    # --- Split mode ---
    def _split_click(self, pos: QPointF) -> None:
        """Record a split point on the polygon boundary."""
        pts = self._current_ai_polygon_points
        n = len(pts)
        # Find closest edge
        best_dist, best_t, best_i = float('inf'), 0.0, -1
        for i in range(n):
            j = (i + 1) % n
            p1, p2 = pts[i], pts[j]
            dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
            L2 = dx*dx + dy*dy
            if L2 < 1:
                d = ((pos.x()-p1.x())**2 + (pos.y()-p1.y())**2)**0.5
            else:
                t = max(0.0, min(1.0, ((pos.x()-p1.x())*dx + (pos.y()-p1.y())*dy) / L2))
                proj_x, proj_y = p1.x() + t*dx, p1.y() + t*dy
                d = ((pos.x()-proj_x)**2 + (pos.y()-proj_y)**2)**0.5
            if d < best_dist and d < 25:
                best_dist, best_i = d, i
                if L2 >= 1:
                    best_t = max(0.0, min(1.0, ((pos.x()-p1.x())*dx + (pos.y()-p1.y())*dy) / L2))
                else:
                    best_t = 0.5

        if best_i < 0:
            return
        # Calculate intersection point
        p1, p2 = pts[best_i], pts[(best_i+1)%n]
        ix = p1.x() + best_t * (p2.x() - p1.x())
        iy = p1.y() + best_t * (p2.y() - p1.y())
        split_pt = QPointF(ix, iy)

        # Mark point
        marker = QGraphicsEllipseItem(ix-5, iy-5, 10, 10)
        marker.setPen(QPen(QColor(255, 255, 0), 2))
        marker.setBrush(QBrush(QColor(255, 255, 0, 120)))
        marker.setZValue(10)
        self.scene.addItem(marker)
        self._split_markers.append(marker)
        self._split_points.append((split_pt, best_i, best_t))

        if len(self._split_points) == 2:
            self._execute_split()

    def _execute_split(self) -> None:
        """Split the selected polygon into two along the split line."""
        pts = self._current_ai_polygon_points
        n = len(pts)
        # Clean up markers
        for m in self._split_markers:
            try: self.scene.removeItem(m)
            except: pass
        self._split_markers = []
        self._split_mode = False

        if len(self._split_points) != 2:
            return

        # Sort split points by edge index
        sp = sorted(self._split_points, key=lambda x: x[1])
        (pt_a, i_a, t_a), (pt_b, i_b, t_b) = sp

        if i_a == i_b:
            return  # Same edge, can't split meaningfully

        # Build two vertex lists by walking around the polygon
        # Path 1: i_a -> i_b (through the split points)
        # Path 2: i_b -> i_a (the other way around)
        def build_path(start_i, start_t, end_i, end_t, go_forward=True):
            path = [QPointF(
                pts[start_i].x() + start_t * (pts[(start_i+1)%n].x() - pts[start_i].x()),
                pts[start_i].y() + start_t * (pts[(start_i+1)%n].y() - pts[start_i].y())
            )]
            cur = (start_i + 1) % n if go_forward else start_i
            target = end_i if go_forward else (end_i + 1) % n
            while cur != target:
                path.append(QPointF(pts[cur].x(), pts[cur].y()))
                cur = (cur + 1) % n if go_forward else (cur - 1 + n) % n
            end_pt = QPointF(
                pts[end_i].x() + end_t * (pts[(end_i+1)%n].x() - pts[end_i].x()),
                pts[end_i].y() + end_t * (pts[(end_i+1)%n].y() - pts[end_i].y())
            )
            path.append(end_pt)
            return path

        poly1 = build_path(i_a, t_a, i_b, t_b, True)
        poly2 = build_path(i_b, t_b, i_a, t_a, True)

        if len(poly1) < 3 or len(poly2) < 3:
            return

        # Delete original polygon and add two new ones
        idx = self._selected_ai_plane_index
        self._delete_plane_quiet(idx)

        for poly in [poly1, poly2]:
            qpoly = QPolygonF(poly)
            color = self._ai_plane_colors[len(self._ai_planes) % len(self._ai_plane_colors)]
            pen = QPen(color, 2)
            poly_item = QGraphicsPolygonItem(qpoly)
            poly_item.setPen(pen)
            poly_item.setBrush(Qt.NoBrush)
            poly_item.setZValue(1)
            self.scene.addItem(poly_item)
            self._ai_overlay_items.append(poly_item)
            self._ai_planes.append({
                'polygon_item': poly_item,
                'polygon_points': poly,
                'color': color,
            })

        self._selected_ai_plane_index = -1
        self._ai_polygon_item = None
        self._current_ai_polygon_points = []
        for v in self._ai_vertex_items:
            try: self.scene.removeItem(v)
            except: pass
        self._ai_vertex_items.clear()
        for lb in self._ai_vertex_label_items:
            try: self.scene.removeItem(lb)
            except: pass
        self._ai_vertex_label_items.clear()
        self._update_outer_perimeter()


    def _reclassify_edge_at(self, pos: QPointF) -> None:
        """Shift+click: find nearest edge of selected polygon and cycle its class."""
        pts = self._current_ai_polygon_points
        n = len(pts)
        best_dist, best_i = float('inf'), -1
        for i in range(n):
            j = (i + 1) % n
            p1, p2 = pts[i], pts[j]
            dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
            L2 = dx*dx + dy*dy
            if L2 < 1:
                d = ((pos.x()-p1.x())**2 + (pos.y()-p1.y())**2)**0.5
            else:
                t = max(0.0, min(1.0, ((pos.x()-p1.x())*dx + (pos.y()-p1.y())*dy) / L2))
                proj_x, proj_y = p1.x() + t*dx, p1.y() + t*dy
                d = ((pos.x()-proj_x)**2 + (pos.y()-proj_y)**2)**0.5
            if d < best_dist and d < 30:
                best_dist, best_i = d, i

        if best_i < 0:
            return

        # Build edge key (normalized: smaller point first)
        i, j = best_i, (best_i + 1) % n
        p1, p2 = pts[i], pts[j]
        x1, y1, x2, y2 = p1.x(), p1.y(), p2.x(), p2.y()
        if x1 < x2 or (x1 == x2 and y1 < y2):
            key = (x1, y1, x2, y2)
        else:
            key = (x2, y2, x1, y1)

        # Count owners for this edge across ALL planes
        from collections import defaultdict
        all_ai = self._ai_planes
        owners = []
        for pi, ap in enumerate(all_ai):
            apts = ap.get("polygon_points", [])
            if len(apts) < 3:
                continue
            for vi in range(len(apts)):
                vj = (vi + 1) % len(apts)
                pa, pb = apts[vi], apts[vj]
                xa, ya = pa.x(), pa.y()
                xb, yb = pb.x(), pb.y()
                if xa < xb or (xa == xb and ya < yb):
                    ek = (xa, ya, xb, yb)
                else:
                    ek = (xb, yb, xa, ya)
                if ek == key:
                    owners.append((pi, pa, pb))

        num_owners = len(owners)

        # Determine current class
        current = self._edge_class_overrides.get(key)
        if not current:
            # Re-run classification to get the auto-detected class
            centroids_test = {}
            for pi, ap in enumerate(all_ai):
                apts = ap.get("polygon_points", [])
                if len(apts) < 3:
                    continue
                cx = sum(p.x() for p in apts) / len(apts)
                cy = sum(p.y() for p in apts) / len(apts)
                centroids_test[pi] = (cx, cy)
            centroids_for_edge = [centroids_test[pi] for pi, _, _ in owners]
            all_pts = [(p.x(), p.y()) for ap in all_ai for p in ap.get("polygon_points", [])]
            hull_edges = self._compute_convex_hull_edges(all_pts)
            okap_dir = getattr(self, '_okap_dir', None)
            if not okap_dir:
                # Fallback: compute on the fly
                edge_planes_full = defaultdict(list)
                for pi, ap in enumerate(all_ai):
                    apts = ap.get("polygon_points", [])
                    if len(apts) < 3:
                        continue
                    for vi in range(len(apts)):
                        vj = (vi + 1) % len(apts)
                        pa, pb = apts[vi], apts[vj]
                        xa, ya = pa.x(), pa.y()
                        xb, yb = pb.x(), pb.y()
                        if xa < xb or (xa == xb and ya < yb):
                            ek = (xa, ya, xb, yb)
                        else:
                            ek = (xb, yb, xa, ya)
                        edge_planes_full[ek].append((pi, pa, pb))
                okap_dir = self._compute_dominant_okap_direction(edge_planes_full, hull_edges)
            current, _, _ = self._classify_edge(p1, p2, centroids_for_edge, hull_edges, key,
                                                 okap_dir, owners, all_ai)
            if not current:
                current = "internal"

        # Cycle order based on edge ownership
        if num_owners <= 1:
            cycle = ["okap", "stit", "internal"]
        else:
            cycle = ["hreben", "narozie", "uzlabie", "internal"]

        try:
            next_idx = (cycle.index(current) + 1) % len(cycle)
        except ValueError:
            next_idx = 0
        self._edge_class_overrides[key] = cycle[next_idx]
        print(f"Edge {best_i} (owners={num_owners}): {current} -> {cycle[next_idx]}")

        # Redraw
        self._update_outer_perimeter()

    def _delete_plane_quiet(self, idx: int) -> None:
        """Remove a plane by index - called during split (don't rebuild perimeter)."""
        if idx < 0 or idx >= len(self._ai_planes):
            return
        plane = self._ai_planes[idx]
        try:
            if 'polygon_item' in plane and plane['polygon_item']:
                self.scene.removeItem(plane['polygon_item'])
        except: pass
        del self._ai_planes[idx]


    def _update_edge_labels(self) -> None:
        """Update edge length labels for the selected polygon."""
        n = len(self._ai_edge_label_items)
        pts = self._current_ai_polygon_points
        if n != len(pts):
            return
        s = self._ai_overlay_px_per_m
        for i in range(n):
            j = (i + 1) % n
            p1, p2 = pts[i], pts[j]
            mx = (p1.x() + p2.x()) / 2
            my = (p1.y() + p2.y()) / 2
            length_m = round(((p2.x()-p1.x())**2 + (p2.y()-p1.y())**2)**0.5 / max(s, 0.1), 1)
            self._ai_edge_label_items[i].setText(f"{length_m}m")
            self._ai_edge_label_items[i].setPos(mx + 3, my - 12)

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
        for el in self._ai_edge_label_items:
            try: self.scene.removeItem(el)
            except: pass
        self._ai_edge_label_items.clear()
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
        pts = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
        self._recalc_dimensions(pts)
        # Highlight selected polygon
        try:
            highlight_pen = QPen(QColor(255, 255, 0), 3)
            self._ai_polygon_item.setPen(highlight_pen)
        except Exception:
            pass
        # Create vertex items and labels for selected polygon
        point_size = max(12.0, 16.0 / max(0.1, self._zoom_factor))
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

        # Create interactive edge labels (shown only for selected polygon)
        for el in self._ai_edge_label_items:
            try: self.scene.removeItem(el)
            except: pass
        self._ai_edge_label_items.clear()
        n_pts = len(self._current_ai_polygon_points)
        s = self._ai_overlay_px_per_m
        for i in range(n_pts):
            j = (i + 1) % n_pts
            p1 = self._current_ai_polygon_points[i]
            p2 = self._current_ai_polygon_points[j]
            mx = (p1.x() + p2.x()) / 2
            my = (p1.y() + p2.y()) / 2
            length_m = round(((p2.x()-p1.x())**2 + (p2.y()-p1.y())**2)**0.5 / max(s, 0.1), 1)
            label = QGraphicsSimpleTextItem(f"{length_m}m")
            label.setBrush(QBrush(QColor(255, 255, 100)))
            label.setZValue(3)
            label.setPos(mx + 3, my - 12)
            self.scene.addItem(label)
            self._ai_edge_label_items.append(label)

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

        # Add-plane mode: handle clicks
        if getattr(self, '_add_plane_mode', False):
            if event.button() == Qt.MouseButton.LeftButton:
                self._add_plane_points.append(scene_pos)
                pt = QGraphicsEllipseItem(scene_pos.x()-4, scene_pos.y()-4, 8, 8)
                pt.setPen(QPen(QColor(255,255,0), 2))
                pt.setBrush(QBrush(QColor(255,255,0,120)))
                pt.setZValue(10)
                self.scene.addItem(pt)
                self._add_plane_items.append(pt)
                if len(self._add_plane_points) >= 2:
                    p1 = self._add_plane_points[-2]
                    p2 = self._add_plane_points[-1]
                    line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
                    line.setPen(QPen(QColor(255,255,0), 2))
                    line.setZValue(9)
                    self.scene.addItem(line)
                    self._add_plane_items.append(line)
            elif event.button() == Qt.MouseButton.RightButton:
                self._finalize_add_plane()
            return

        # Check if click is on an AI polygon (select it)
        if event.button() == Qt.MouseButton.LeftButton and self._ai_planes and self._ai_overlay_active:
            pt = QPointF(scene_pos.x(), scene_pos.y())
            hit_idx = -1
            for idx in range(len(self._ai_planes) - 1, -1, -1):  # top-first
                poly = self._ai_planes[idx].get("polygon_points", [])
                if poly and len(poly) >= 3:
                    qpf = QPolygonF(poly)
                    if qpf.containsPoint(pt, Qt.FillRule.OddEvenFill):
                        hit_idx = idx
                        break
            if hit_idx >= 0:
                # Edge reclassification: Shift+click near edge of selected polygon
                if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier and
                    hit_idx == self._selected_ai_plane_index):
                    self._reclassify_edge_at(scene_pos)
                    return
                # Split mode: click on boundary of selected polygon
                if (getattr(self, '_split_mode', False) and
                    hit_idx == self._selected_ai_plane_index):
                    self._split_click(scene_pos)
                    return
                # Merge mode: merge selected with clicked polygon
                if getattr(self, '_merge_mode', False) and hit_idx != self._selected_ai_plane_index:
                    self._merge_polygons(self._selected_ai_plane_index, hit_idx)
                    self._merge_mode = False
                    return
                self._select_ai_plane(hit_idx)
                self.ai_overlay_plane_selected.emit(hit_idx, [(float(p.x()), float(p.y())) for p in self._ai_planes[hit_idx]["polygon_points"]])
                return  # consume event, no panning

        # Default panning behavior if no other mode is active
        if event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._last_pan_pos = event.position()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        """Delete vertex (Delete) or entire polygon (Shift+Delete)."""
        # Shift+Delete: delete entire selected polygon
        if (event.key() == Qt.Key.Key_Delete and
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier and
            self._selected_ai_plane_index >= 0):
            self._delete_selected_polygon()
            return
        # Shift+N: toggle add-plane drawing mode
        if (event.key() == Qt.Key.Key_N and
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._toggle_add_plane_mode()
            return
        # Shift+M: merge mode - click second polygon to merge with selected
        if (event.key() == Qt.Key.Key_M and
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier and
            self._selected_ai_plane_index >= 0):
            self._merge_mode = not getattr(self, '_merge_mode', False)
            print(f"Merge mode: {'ON' if self._merge_mode else 'OFF'}")
            return
        # Shift+S: split mode - click two points on polygon boundary
        if (event.key() == Qt.Key.Key_S and
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier and
            self._selected_ai_plane_index >= 0):
            self._split_mode = not getattr(self, '_split_mode', False)
            self._split_points = []
            self._split_markers = []
            print(f"Split mode: {'ON' if self._split_mode else 'OFF'}")
            return
        # Delete: delete closest vertex
        if event.key() == Qt.Key.Key_Delete and self._ai_vertex_items and self._selected_ai_plane_index >= 0:
            n = len(self._current_ai_polygon_points)
            if n <= 3:
                super().keyPressEvent(event)
                return
            # Find vertex closest to mouse cursor
            cursor_pos = self.mapToScene(self.mapFromGlobal(self.cursor().pos()))
            min_dist = float('inf')
            min_idx = -1
            for i, pt in enumerate(self._current_ai_polygon_points):
                d = ((pt.x() - cursor_pos.x())**2 + (pt.y() - cursor_pos.y())**2)**0.5
                if d < min_dist:
                    min_dist = d
                    min_idx = i
            if min_dist < 30 and min_idx >= 0:
                del self._current_ai_polygon_points[min_idx]
                new_poly = QPolygonF(self._current_ai_polygon_points)
                self._ai_polygon_item.setPolygon(new_poly)
                if min_idx < len(self._ai_vertex_items):
                    self.scene.removeItem(self._ai_vertex_items[min_idx])
                    del self._ai_vertex_items[min_idx]
                if min_idx < len(self._ai_vertex_label_items):
                    self.scene.removeItem(self._ai_vertex_label_items[min_idx])
                    del self._ai_vertex_label_items[min_idx]
                for j, v in enumerate(self._ai_vertex_items):
                    v._index = j
                pts = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
                self._recalc_dimensions(pts)
                self._sync_edited_polygon_to_planes()
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click on polygon edge to add a new vertex at midpoint."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        if not self._ai_planes or not self._ai_overlay_active or self._selected_ai_plane_index < 0:
            super().mouseDoubleClickEvent(event)
            return

        scene_pos = self.mapToScene(event.position().toPoint())
        pts = self._current_ai_polygon_points
        n = len(pts)
        if n < 2:
            super().mouseDoubleClickEvent(event)
            return

        best_dist = float('inf')
        best_idx = -1
        best_mid = QPointF()
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
            seg_len_sq = dx*dx + dy*dy
            if seg_len_sq < 1:
                continue
            t = max(0.0, min(1.0, ((scene_pos.x()-p1.x())*dx + (scene_pos.y()-p1.y())*dy) / seg_len_sq))
            proj = QPointF(p1.x() + t*dx, p1.y() + t*dy)
            d = ((scene_pos.x()-proj.x())**2 + (scene_pos.y()-proj.y())**2)**0.5
            if d < best_dist and d < 25:
                best_dist = d
                best_idx = i
                best_mid = proj

        if best_idx >= 0:
            self._current_ai_polygon_points.insert(best_idx + 1, best_mid)
            new_poly = QPolygonF(self._current_ai_polygon_points)
            self._ai_polygon_item.setPolygon(new_poly)
            self._sync_edited_polygon_to_planes()
            self._select_ai_plane(self._selected_ai_plane_index)
            pts_list = [(p.x(), p.y()) for p in self._current_ai_polygon_points]
            self._recalc_dimensions(pts_list)
            return

        super().mouseDoubleClickEvent(event)

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