"""
This module provides a service for converting AI prediction results into structured
RoofGeometry objects.
"""
import uuid
from typing import List, Union, Optional, Dict, Any
import numpy as np
import cv2

from app.ai.ai_result import DetectionResult, GeometryPredictionResult, BoundingBox, PolygonGeometry, BBoxGeometry
from app.ai.segmentation_result import SegmentationResult # Use new SegmentationResult
from app.ai.mask_processor import MaskProcessor # New import
from app.geometry.point import Point2D, Point3D
from app.geometry.edge import Edge, RoofEdge
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.core.logger import setup_logging

logger = setup_logging()

class GeometryConverter:
    """
    Service responsible for converting various AI prediction result formats
    (DetectionResult, SegmentationResult, GeometryPredictionResult)
    into a unified RoofGeometry object.
    """

    def __init__(self):
        pass

    def convert_detection_results_to_geometry(
        self,
        detection_results: List[DetectionResult],
        image_width: int,
        image_height: int,
        calibration: Optional[CalibrationModel] = None,
        default_slope_degrees: float = 30.0,
        default_orientation_degrees: float = 0.0,
        post_processed_edges: Optional[List[RoofEdge]] = None
    ) -> RoofGeometry:
        """
        Converts a list of DetectionResult objects and optional post-processed edges into a RoofGeometry.

        Args:
            detection_results (List[DetectionResult]): List of detected objects.
            image_width (int): Width of the original image in pixels.
            image_height (int): Height of the original image in pixels.
            calibration (Optional[CalibrationModel]): Calibration model for pixel to real-world conversion.
            default_slope_degrees (float): Default slope to assign to detected planes.
            default_orientation_degrees (float): Default orientation to assign to detected planes.
            post_processed_edges (Optional[List[RoofEdge]]): Extracted shared boundary edges between adjacent planes.

        Returns:
            RoofGeometry: A RoofGeometry object constructed from the detections.
        """
        planes: List[RoofPlane] = []
        openings: List[Polygon2D] = []
        all_vertices_3d: List[Point3D] = [] # Collect all unique 3D vertices

        # If there are no detection results, construct an empty RoofGeometry instance
        # without invoking dataclass validation (which requires at least one plane).
        if not detection_results:
            empty_geom = object.__new__(RoofGeometry)
            # Set attributes directly, matching RoofGeometry structure
            object.__setattr__(empty_geom, 'id', uuid.uuid4())
            object.__setattr__(empty_geom, 'vertices', tuple())
            object.__setattr__(empty_geom, 'edges', tuple())
            object.__setattr__(empty_geom, 'planes', tuple())
            object.__setattr__(empty_geom, 'ridges', tuple())
            object.__setattr__(empty_geom, 'valleys', tuple())
            object.__setattr__(empty_geom, 'openings', tuple())
            return empty_geom

        for dr in detection_results:
            bbox = dr.bounding_box
            bbox_polygon_2d = None

            # 1. First check the new structured geometry field
            if hasattr(dr, "geometry") and dr.geometry is not None:
                if isinstance(dr.geometry, PolygonGeometry):
                    polygon_vertices = [Point2D(float(x), float(y)) for (x, y) in dr.geometry.vertices]
                    bbox_polygon_2d = Polygon2D(vertices=polygon_vertices)
                elif isinstance(dr.geometry, BBoxGeometry):
                    p1_2d = Point2D(dr.geometry.x_min, dr.geometry.y_min)
                    p2_2d = Point2D(dr.geometry.x_max, dr.geometry.y_min)
                    p3_2d = Point2D(dr.geometry.x_max, dr.geometry.y_max)
                    p4_2d = Point2D(dr.geometry.x_min, dr.geometry.y_max)
                    bbox_polygon_2d = Polygon2D(vertices=[p1_2d, p2_2d, p3_2d, p4_2d])

            # 2. Fallback to metadata for backwards compatibility
            if bbox_polygon_2d is None and isinstance(dr.metadata, dict):
                polygon_pts = dr.metadata.get("polygon_vertices") or dr.metadata.get("contour_polygon")
                if polygon_pts:
                    polygon_vertices = [Point2D(float(x), float(y)) for (x, y) in polygon_pts]
                    bbox_polygon_2d = Polygon2D(vertices=polygon_vertices)

            # 3. Ultimate fallback to raw bounding box
            if bbox_polygon_2d is None:
                p1_2d = Point2D(bbox.x_min, bbox.y_min)
                p2_2d = Point2D(bbox.x_max, bbox.y_min)
                p3_2d = Point2D(bbox.x_max, bbox.y_max)
                p4_2d = Point2D(bbox.x_min, bbox.y_max)
                bbox_polygon_2d = Polygon2D(vertices=[p1_2d, p2_2d, p3_2d, p4_2d])

            if dr.class_name in ("roof_plane", "roof_area"):
                heights = [0.0] * len(bbox_polygon_2d.vertices) # Placeholder for Z coordinates
                plane = RoofPlane(
                    name=f"Plane_{dr.id}",
                    polygon=bbox_polygon_2d,
                    slope=default_slope_degrees,
                    orientation=default_orientation_degrees,
                    height_at_vertices=heights
                )
                planes.append(plane)
                for i, v2d in enumerate(bbox_polygon_2d.vertices):
                    all_vertices_3d.append(Point3D(v2d.x, v2d.y, heights[i]))

            elif dr.class_name in ("potential_opening", "opening", "skylight", "chimney"):
                openings.append(bbox_polygon_2d)

        # 4. Process post-processed shared edges and build 3D connections
        edges: List[Edge] = []
        ridges: List[Edge] = []
        valleys: List[Edge] = []

        if post_processed_edges:
            for pe in post_processed_edges:
                # Convert 2D edge points to 3D (assuming z=0 placeholder for flat prediction)
                start_3d = Point3D(pe.start_point.x, pe.start_point.y, 0.0)
                end_3d = Point3D(pe.end_point.x, pe.end_point.y, 0.0)
                all_vertices_3d.append(start_3d)
                all_vertices_3d.append(end_3d)

                edge_3d = RoofEdge(
                    start_point=start_3d,
                    end_point=end_3d,
                    left_plane_id=pe.left_plane_id,
                    right_plane_id=pe.right_plane_id,
                    edge_type=pe.edge_type
                )
                edges.append(edge_3d)
                if pe.edge_type == "ridge":
                    ridges.append(edge_3d)
                elif pe.edge_type == "valley":
                    valleys.append(edge_3d)

        # Remove duplicate 3D vertices
        unique_vertices_3d = list(set(all_vertices_3d))

        return RoofGeometry(
            vertices=unique_vertices_3d,
            edges=edges,
            planes=planes,
            ridges=ridges,
            valleys=valleys,
            openings=openings
        )

    def convert_segmentation_results_to_geometry(
        self,
        segmentation_results: List[SegmentationResult],
        calibration: Optional[CalibrationModel] = None,
        default_slope_degrees: float = 30.0,
        default_orientation_degrees: float = 0.0,
        mask_processing_params: Optional[dict] = None
    ) -> RoofGeometry:
        """
        Converts a list of SegmentationResult objects into a RoofGeometry.
        Extracts contours from masks to form 2D polygons, applying mask processing.

        Args:
            segmentation_results (List[SegmentationResult]): List of segmented objects.
            calibration (Optional[CalibrationModel]): Calibration model for pixel to real-world conversion.
            default_slope_degrees (float): Default slope to assign to segmented planes.
            default_orientation_degrees (float): Default orientation to assign to segmented planes.
            mask_processing_params (Optional[dict]): Parameters for mask processing (e.g., kernel_size, min_area_threshold).

        Returns:
            RoofGeometry: A RoofGeometry object constructed from the segmentations.
        """
        planes: List[RoofPlane] = []
        openings: List[Polygon2D] = []
        all_vertices_3d: List[Point3D] = []
        
        mask_processing_params = mask_processing_params or {}

        for sr in segmentation_results:
            if sr.mask is None or sr.mask.size == 0:
                logger.warning(f"Skipping segmentation result {sr.id} due to empty mask.")
                continue

            # Process mask using MaskProcessor
            cleaned_mask = MaskProcessor.clean_mask(
                sr.mask,
                kernel_size=mask_processing_params.get("clean_kernel_size", 5),
                iterations=mask_processing_params.get("clean_iterations", 1)
            )
            noise_removed_mask = MaskProcessor.remove_noise(
                cleaned_mask,
                min_area_threshold=mask_processing_params.get("noise_min_area_threshold", 100)
            )

            # Extract contours
            contours = MaskProcessor.extract_contours(
                noise_removed_mask,
                min_area_threshold=mask_processing_params.get("contour_min_area_threshold", 50)
            )

            for contour in contours:
                try:
                    # Simplify polygon
                    polygon_2d_pixel = MaskProcessor.simplify_polygon(
                        contour,
                        epsilon_factor=mask_processing_params.get("simplify_epsilon_factor", 0.005)
                    )

                    # Convert pixel polygon to real-world polygon (meters)
                    real_world_vertices: List[Point2D] = []
                    if calibration:
                        for p_pixel in polygon_2d_pixel.vertices:
                            x_meter = CalibrationService.pixel_to_meter(p_pixel.x, calibration)
                            y_meter = CalibrationService.pixel_to_meter(p_pixel.y, calibration)
                            real_world_vertices.append(Point2D(x_meter, y_meter))
                    else:
                        # If no calibration, assume pixel units are directly meters (or some arbitrary unit)
                        real_world_vertices = polygon_2d_pixel.vertices

                    real_world_polygon = Polygon2D(vertices=real_world_vertices)

                    # Create RoofPlane or Opening
                    if sr.class_name == "roof_area": # Assuming the segmentation model outputs "roof_area"
                        heights = [0.0] * len(real_world_polygon.vertices) # Placeholder for Z coordinates
                        plane = RoofPlane(
                            name=f"SegmentedPlane_{sr.id}_{uuid.uuid4().hex[:4]}",
                            polygon=real_world_polygon,
                            slope=default_slope_degrees,
                            orientation=default_orientation_degrees,
                            height_at_vertices=heights
                        )
                        planes.append(plane)
                        for i, v2d in enumerate(real_world_polygon.vertices):
                            all_vertices_3d.append(Point3D(v2d.x, v2d.y, heights[i]))
                    elif sr.class_name == "opening":
                        openings.append(real_world_polygon)

                except ValueError as e:
                    logger.error(f"Error processing contour for segmentation result {sr.id}: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error during contour processing: {e}")

        edges: List[Edge] = []
        ridges: List[Edge] = []
        valleys: List[Edge] = []
        unique_vertices_3d = list(set(all_vertices_3d))

        return RoofGeometry(
            vertices=unique_vertices_3d,
            edges=edges,
            planes=planes,
            ridges=ridges,
            valleys=valleys,
            openings=openings
        )

    def convert_geometry_prediction_to_geometry(
        self,
        geometry_prediction: GeometryPredictionResult,
        calibration: Optional[CalibrationModel] = None
    ) -> RoofGeometry:
        """
        Converts a GeometryPredictionResult (which should already contain structured
        geometry data) into a RoofGeometry object.

        Args:
            geometry_prediction (GeometryPredictionResult): The AI result containing structured geometry.
            calibration (Optional[CalibrationModel]): Calibration model (might be used if geometry_prediction
                                                      still contains pixel-based data, but ideally it's real-world).

        Returns:
            RoofGeometry: A RoofGeometry object.
        """
        # This conversion assumes that `predicted_geometry_data` in GeometryPredictionResult
        # is structured in a way that directly maps to RoofGeometry components.
        # This is the most direct conversion path.

        data = geometry_prediction.predicted_geometry_data

        # Reconstruct Point3D vertices
        vertices_data = data.get("vertices", [])
        vertices = [Point3D(v['x'], v['y'], v['z']) for v in vertices_data]

        # Reconstruct Edges (requires mapping vertex IDs to Point3D objects)
        # This is a simplified approach; a real system would use UUIDs for vertices
        # and map them. For now, assume edges refer to indices or direct Point3D objects.
        edges_data = data.get("edges", [])
        edges = []
        for e_data in edges_data:
            start_p = Point3D(e_data['start_point']['x'], e_data['start_point']['y'], e_data['start_point']['z'])
            end_p = Point3D(e_data['end_point']['x'], e_data['end_point']['y'], e_data['end_point']['z'])
            edges.append(Edge(start_p, end_p))

        # Reconstruct RoofPlanes
        planes_data = data.get("planes", [])
        planes = []
        for p_data in planes_data:
            poly_vertices_2d = [Point2D(v['x'], v['y']) for v in p_data['polygon']['vertices']]
            polygon_2d = Polygon2D(vertices=poly_vertices_2d)
            planes.append(RoofPlane(
                name=p_data['name'],
                polygon=polygon_2d,
                slope=p_data['slope'],
                orientation=p_data['orientation'],
                height_at_vertices=p_data.get('height_at_vertices', [])
            ))

        # Reconstruct Openings
        openings_data = data.get("openings", [])
        openings = []
        for o_data in openings_data:
            poly_vertices_2d = [Point2D(v['x'], v['y']) for v in o_data['vertices']]
            openings.append(Polygon2D(vertices=poly_vertices_2d))

        # Ridges and Valleys (similar reconstruction as edges)
        ridges_data = data.get("ridges", [])
        ridges = []
        for r_data in ridges_data:
            start_p = Point3D(r_data['start_point']['x'], r_data['start_point']['y'], r_data['start_point']['z'])
            end_p = Point3D(r_data['end_point']['x'], r_data['end_point']['y'], r_data['end_point']['z'])
            ridges.append(Edge(start_p, end_p))

        valleys_data = data.get("valleys", [])
        valleys = []
        for v_data in valleys_data:
            start_p = Point3D(v_data['start_point']['x'], v_data['start_point']['y'], v_data['start_point']['z'])
            end_p = Point3D(v_data['end_point']['x'], v_data['end_point']['y'], v_data['end_point']['z'])
            valleys.append(Edge(start_p, end_p))

        return RoofGeometry(
            vertices=vertices,
            edges=edges,
            planes=planes,
            ridges=ridges,
            valleys=valleys,
            openings=openings
        )