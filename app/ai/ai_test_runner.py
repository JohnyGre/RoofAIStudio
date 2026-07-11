"""
This module provides a command-line utility to run AI segmentation tests
on individual images and save the results.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import uuid

from app.ai.ai_engine import AIEngine
from app.ai.models.yolo_adapter import YOLOSegmentationModel
from app.ai.segmentation_model import AbstractSegmentationModel
from app.ai.segmentation_result import SegmentationResult
from app.ai.mask_processor import MaskProcessor
from app.ai.roof_segmentation import RoofSegmentationService
from app.core.image.image_loader import ImageLoader
from app.core.logger import setup_logging
from app.core.app_paths import app_paths
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.geometry.roof_geometry import RoofGeometry
from app.geometry.point import Point2D

logger = setup_logging()

class AITestRunner:
    """
    Runs an AI segmentation test workflow on a given image.
    """

    def __init__(self, output_base_dir: Path):
        self.ai_engine = AIEngine()
        self.output_base_dir = output_base_dir
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AI Test Runner initialized. Output directory: {self.output_base_dir}")

        # Register and load a default YOLO segmentation model for testing
        self.yolo_model_id = uuid.uuid4()
        self.yolo_model = YOLOSegmentationModel(model_id=self.yolo_model_id)
        self.ai_engine.register_model(self.yolo_model)
        
        # Placeholder for YOLO model weights. User needs to provide a real path.
        # For testing, we'll use a dummy path and rely on the adapter's error handling.
        self.yolo_model_path = app_paths.ai_models_dir / "yolov8n-seg.pt" 
        if not self.yolo_model_path.exists():
            logger.warning(f"YOLO model weights not found at {self.yolo_model_path}. "
                           "Please download a YOLOv8 segmentation model (e.g., yolov8n-seg.pt) "
                           "and place it in this directory for full functionality.")
            # Create a dummy file to prevent FileNotFoundError during load, but it won't work
            self.yolo_model_path.touch() 
        
        try:
            self.ai_engine.load_model(model_id=self.yolo_model_id, model_path=str(self.yolo_model_path))
        except Exception as e:
            logger.error(f"Failed to load YOLO model for test runner: {e}")
            logger.warning("AI Test Runner will proceed with a non-functional YOLO model.")

    def run_test(self, image_path: Path, output_name: Optional[str] = None) -> Optional[Path]:
        """
        Executes the full AI segmentation and geometry conversion workflow for a single image.

        Args:
            image_path (Path): Path to the input image.
            output_name (Optional[str]): Base name for output files. If None, uses image filename.

        Returns:
            Optional[Path]: Path to the generated JSON result file, or None if the test fails.
        """
        logger.info(f"Running AI test for image: {image_path}")
        if not image_path.exists():
            logger.error(f"Input image not found: {image_path}")
            return None

        output_name = output_name if output_name else image_path.stem
        test_output_dir = self.output_base_dir / output_name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load test image
        image_data_bgr = ImageLoader.load_image(image_path, as_opencv=True)
        if image_data_bgr is None:
            logger.error(f"Failed to load image data for {image_path}")
            return None
        
        original_h, original_w, _ = image_data_bgr.shape

        # Placeholder calibration: 100 pixels = 1 meter
        # In a real scenario, this would come from user input or metadata
        placeholder_p1 = Point2D(0.0, 0.0)
        placeholder_p2 = Point2D(100.0, 0.0) # 100 pixels
        placeholder_distance_meters = 1.0 # 1 meter
        calibration = CalibrationService.calibrate_from_distance(
            placeholder_p1, placeholder_p2, placeholder_distance_meters
        )

        # 2. Run segmentation and convert to RoofGeometry
        try:
            # Use RoofSegmentationService directly for this test runner
            if not isinstance(self.yolo_model, AbstractSegmentationModel):
                raise TypeError("Registered model is not an AbstractSegmentationModel.")
            
            roof_segmentation_service = RoofSegmentationService(self.yolo_model)
            
            roof_geometry: RoofGeometry = roof_segmentation_service.segment_and_convert_to_geometry(
                image=image_data_bgr,
                image_width=original_w,
                image_height=original_h,
                calibration=calibration,
                default_slope_degrees=30.0,
                default_orientation_degrees=0.0,
                mask_processing_params={
                    "clean_kernel_size": 5,
                    "noise_min_area_threshold": 100,
                    "contour_min_area_threshold": 50,
                    "simplify_epsilon_factor": 0.005
                },
                segmentation_params={
                    "confidence_threshold": 0.25,
                    "iou_threshold": 0.7
                }
            )
        except Exception as e:
            logger.error(f"AI segmentation and geometry conversion failed: {e}")
            return None

        # 3. Save results
        # Save original image
        cv2.imwrite(str(test_output_dir / f"{output_name}_original.jpg"), image_data_bgr)

        # Save segmentation masks (example: combine all masks into one image)
        combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        for plane in roof_geometry.planes:
            # For simplicity, draw the 2D polygon of the plane onto the mask
            # In a real scenario, you'd get the mask directly from SegmentationResult
            if plane.polygon.vertices:
                pts = np.array([[p.x, p.y] for p in plane.polygon.vertices], np.int3).reshape((-1, 1, 2))
                cv2.fillPoly(combined_mask, [pts], 255)
        cv2.imwrite(str(test_output_dir / f"{output_name}_mask.png"), combined_mask)

        # Save extracted contours (example: draw on original image)
        image_with_contours = image_data_bgr.copy()
        for plane in roof_geometry.planes:
            if plane.polygon.vertices:
                pts = np.array([[p.x, p.y] for p in plane.polygon.vertices], np.int3).reshape((-1, 1, 2))
                cv2.polylines(image_with_contours, [pts], True, (0, 255, 0), 2) # Green contours
        cv2.imwrite(str(test_output_dir / f"{output_name}_contours.jpg"), image_with_contours)

        # Save JSON result (simplified RoofGeometry representation)
        geometry_json_path = test_output_dir / f"{output_name}_geometry.json"
        try:
            # Convert RoofGeometry to a serializable dict
            serializable_geometry = {
                "planes": [
                    {
                        "name": p.name,
                        "polygon_vertices": [{"x": v.x, "y": v.y} for v in p.polygon.vertices],
                        "slope": p.slope,
                        "orientation": p.orientation,
                        "true_area_sq_m": p.true_area # Assuming polygon vertices are in meters
                    } for p in roof_geometry.planes
                ],
                "openings": [
                    {
                        "polygon_vertices": [{"x": v.x, "y": v.y} for v in o.vertices],
                        "area_sq_m": o.area # Assuming polygon vertices are in meters
                    } for o in roof_geometry.openings
                ],
                "total_area_sq_m": roof_geometry.calculate_total_area()
            }
            with open(geometry_json_path, 'w') as f:
                json.dump(serializable_geometry, f, indent=4)
            logger.info(f"AI test results saved to {test_output_dir}")
            return geometry_json_path
        except Exception as e:
            logger.error(f"Failed to save JSON geometry result: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description="Run AI segmentation test on an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--output_dir", type=str, default=str(app_paths.data_dir / "ai_test_results"),
                        help="Directory to save test results.")
    parser.add_argument("--output_name", type=str, help="Base name for output files (defaults to image filename).")
    
    args = parser.parse_args()

    output_base_dir = Path(args.output_dir)
    runner = AITestRunner(output_base_dir)
    
    image_path = Path(args.image_path)
    runner.run_test(image_path, args.output_name)

if __name__ == "__main__":
    main()
