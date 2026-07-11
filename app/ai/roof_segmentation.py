"""
This module defines the RoofSegmentationService, orchestrating the workflow
from image input to segmented roof geometry.
"""

from typing import List, Optional, Union
import numpy as np
import uuid

from app.ai.segmentation_model import AbstractSegmentationModel
from app.ai.segmentation_result import SegmentationResult
from app.ai.mask_processor import MaskProcessor
from app.geometry.point import Point2D, Point3D
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.core.logger import setup_logging

logger = setup_logging()

class RoofSegmentationService:
    """
    Service responsible for performing roof segmentation using an AI model
    and converting the resulting masks into structured RoofGeometry.
    """

    def __init__(self, segmentation_model: AbstractSegmentationModel):
        """
        Initializes the RoofSegmentationService with a specific segmentation model.

        Args:
            segmentation_model (AbstractSegmentationModel): The AI model to use for segmentation.
        """
        self._segmentation_model = segmentation_model
        if not self._segmentation_model.is_loaded:
            logger.warning(f"Segmentation model '{self._segmentation_model.model_name}' is not loaded. "
                           "Ensure it's loaded before performing segmentation.")

    def segment_and_convert_to_geometry(
        self,
        image: np.ndarray,
        image_width: int,
        image_height: int,
        calibration: Optional[CalibrationModel] = None,
        default_slope_degrees: float = 30.0,
        default_orientation_degrees: float = 0.0,
        mask_processing_params: Optional[dict] = None,
        segmentation_params: Optional[dict] = None
    ) -> RoofGeometry:
        """
        Performs roof segmentation on an image and converts the resulting masks
        into a RoofGeometry object.

        Args:
            image (np.ndarray): The input image (OpenCV BGR format).
            image_width (int): Width of the original image in pixels.
            image_height (int): Height of the original image in pixels.
            calibration (Optional[CalibrationModel]): Calibration model for pixel to real-world conversion.
            default_slope_degrees (float): Default slope to assign to segmented planes.
            default_orientation_degrees (float): Default orientation to assign to segmented planes.
            mask_processing_params (Optional[dict]): Parameters for mask processing (e.g., kernel_size, min_area_threshold).
            segmentation_params (Optional[dict]): Parameters for the segmentation model's `segment` method.

        Returns:
            RoofGeometry: A RoofGeometry object constructed from the segmented masks.

        Raises:
            RuntimeError: If the segmentation model is not loaded or segmentation fails.
        """
        if not self._segmentation_model.is_loaded:
            raise RuntimeError(f"Segmentation model '{self._segmentation_model.model_name}' is not loaded.")

        logger.info(f"Performing roof segmentation using model: {self._segmentation_model.model_name}")
        
        segmentation_params = segmentation_params or {}
        mask_processing_params = mask_processing_params or {}

        # 1. Perform segmentation
        raw_segmentation_results: List[SegmentationResult] = self._segmentation_model.segment(image, **segmentation_params)

        planes: List[RoofPlane] = []
        openings: List[Polygon2D] = []
        all_vertices_3d: List[Point3D] = []

        for sr in raw_segmentation_results:
            if sr.mask is None or sr.mask.size == 0:
                logger.warning(f"Skipping segmentation result {sr.id} due to empty mask.")
                continue

            # 2. Process mask
            cleaned_mask = MaskProcessor.clean_mask(
                sr.mask,
                kernel_size=mask_processing_params.get("clean_kernel_size", 5),
                iterations=mask_processing_params.get("clean_iterations", 1)
            )
            noise_removed_mask = MaskProcessor.remove_noise(
                cleaned_mask,
                min_area_threshold=mask_processing_params.get("noise_min_area_threshold", 100)
            )

            # 3. Extract contours
            contours = MaskProcessor.extract_contours(
                noise_removed_mask,
                min_area_threshold=mask_processing_params.get("contour_min_area_threshold", 50)
            )

            for contour in contours:
                try:
                    # 4. Simplify polygon
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

                    # 5. Create RoofPlane or Opening
                    if sr.class_name == "roof_area": # Assuming the segmentation model outputs "roof_area"
                        heights = [0.0] * len(real_world_polygon.vertices) # Placeholder for Z coordinates
                        plane = RoofPlane(
                            name=f"SegmentedPlane_{uuid.uuid4().hex[:8]}",
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

        # Construct RoofGeometry
        edges: List[Edge] = [] # Edges, ridges, valleys would be derived from plane intersections
        ridges: List[Edge] = []
        valleys: List[Edge] = []
        unique_vertices_3d = list(set(all_vertices_3d))

        roof_geometry = RoofGeometry(
            vertices=unique_vertices_3d,
            edges=edges,
            planes=planes,
            ridges=ridges,
            valleys=valleys,
            openings=openings
        )
        logger.info("RoofGeometry created from segmentation masks.")
        return roof_geometry
