"""
Roof detector facade - primary entry point for roof analysis.
v0.5.0: + SAMRoofSegmenter (zero-shot, any view angle)
"""
import uuid
from typing import Any, Dict, List, Union, Optional
import cv2
import numpy as np

from app.ai.models.vision_detector import VisionDetector
from app.ai.ai_result import DetectionResult, SegmentationResult, GeometryPredictionResult
from app.core.logger import setup_logging
from app.ai.hybrid_roof_detector import HybridRoofDetector
from app.ai.sam_roof_segmenter import SAMRoofSegmenter
from app.ai.pipeline.roof_line_detector import RoofLineDetector
from app.ai.pipeline.roof_feature_detector import RoofFeatureDetector
from app.ai.pipeline.roof_damage_detector import RoofDamageDetector

logger = setup_logging()

class RoofDetector(VisionDetector):
    """
    Primary roof detector orchestrating:
    - HybridRoofDetector (YOLO segmentation + multi-strategy OpenCV + color/texture filter)
    - SAMRoofSegmenter (zero-shot SAM, any view angle - street/drone/satellite)
    - RoofLineDetector (Hough-based ridge/valley/eave detection)
    - RoofFeatureDetector (skylights, chimneys)
    - RoofDamageDetector (cracks, missing shingles)
    """

    MODEL_NAME = "OpenCV_Roof_Detector"
    VERSION = "0.5.0-sam"

    def __init__(self, model_id: Optional[uuid.UUID] = None):
        super().__init__(self.MODEL_NAME, self.VERSION, model_id)
        self.hybrid_detector = HybridRoofDetector()
        self.sam_segmenter: Optional[SAMRoofSegmenter] = None
        self.line_detector = RoofLineDetector()
        self.feature_detector = RoofFeatureDetector()
        self.damage_detector = RoofDamageDetector()

        self._model_info: Dict[str, Any] = {
            "name": self.MODEL_NAME,
            "version": self.VERSION,
            "description": (
                "Hybrid roof detection: YOLOv8-seg + multi-strategy OpenCV + "
                "SAM zero-shot segmentation (any view angle) + "
                "Hough line detection + color/texture classifier."
            ),
            "input_requirements": "BGR NumPy array image",
            "output_format": "List[DetectionResult]",
            "sub_detectors": ["planes", "sam", "lines", "features", "damage"],
        }

    def load(self, model_path: str = "", device: str = "cpu", **kwargs) -> None:
        """Load the detector. Auto-loads YOLO and optionally SAM."""
        yolo_path = kwargs.get("yolo_model_path", "ai_models/yolov8n_building_seg.pt")
        loaded = self.hybrid_detector.load_yolo(yolo_path)
        logger.info(
            f"RoofDetector v{self.VERSION} loaded. YOLO: {'yes' if loaded else 'no (OpenCV only)'}"
        )

        # Load SAM if available
        load_sam = kwargs.get("enable_sam", True)
        if load_sam:
            try:
                import os
                sam_checkpoint = kwargs.get(
                    "sam_checkpoint", "ai_models/sam_vit_b_01ec64.pth"
                )
                if os.path.exists(sam_checkpoint):
                    self.sam_segmenter = SAMRoofSegmenter(
                        model_type="vit_b",
                        yolo_model_path=kwargs.get(
                            "sam_yolo_path", "ai_models/roof_gmaps_v2.pt"
                        ),
                        device=device,
                    )
                    self.sam_segmenter.load(sam_checkpoint=sam_checkpoint)
                    logger.info("SAM segmenter loaded")
                else:
                    logger.info(f"SAM checkpoint not found at {sam_checkpoint}, skipping SAM")
            except Exception as e:
                logger.warning(f"SAM load failed: {e}. SAM disabled.")

        self._is_loaded = True

    def detect(
        self, image: np.ndarray, **kwargs
    ) -> List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]]:
        """
        Run detection on the image.

        Args:
            image: BGR numpy array
            **kwargs:
                - enable_planes (default True) - hybrid roof plane detection
                - enable_sam (default True, if SAM loaded) - SAM zero-shot refinement
                - enable_lines (default True) - Hough line detection
                - enable_features (default False)
                - enable_damage (default False)
                - use_yolo (default True) - enable YOLO in hybrid detector
                - use_opencv (default True) - enable OpenCV in hybrid detector
                - max_results (default 15) - max roof planes
                - min_roof_score (default 0.25) - minimum roof-likeness score
                - yolo_conf (default 0.2) - YOLO confidence threshold

        Returns:
            Combined list of DetectionResult
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model '{self.model_name}' is not loaded. Call load() first.")
        if not self.validate(image):
            raise ValueError("Invalid image input.")

        all_results: List[Union[DetectionResult, SegmentationResult, GeometryPredictionResult]] = []

        # 1. Broad-phase roof region detection using OpenCV
        if kwargs.get("enable_planes", True):
            logger.debug("Running HybridRoofDetector (OpenCV only) for broad-phase detection...")
            # Force OpenCV-only for the initial pass
            opencv_kwargs = kwargs.copy()
            opencv_kwargs['use_yolo'] = False
            opencv_kwargs['use_opencv'] = True
            
            roof_regions = self.hybrid_detector.detect(
                image,
                **opencv_kwargs
            )
            all_results.extend(roof_regions)
            logger.info(f"RoofDetector: Found {len(roof_regions)} broad roof regions with OpenCV.")

            # 2. Fine-grained polygon detection within each roof region using SAM/YOLO
            if kwargs.get("enable_sam", True) and self.sam_segmenter is not None:
                logger.debug("Running SAM zero-shot refinement within each broad roof region...")
                for region in roof_regions:
                    # Extract the bounding box of the detected region
                    x_min, y_min, x_max, y_max = region.geometry.x_min, region.geometry.y_min, region.geometry.x_max, region.geometry.y_max
                    
                    # Crop the image to the detected region
                    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]

                    if cropped_image.size == 0:
                        continue

                    # Run SAM on the cropped image
                    sam_results = self.sam_segmenter.segment_auto(
                        cropped_image,
                        conf=kwargs.get("sam_conf", 0.25),
                        imgsz=kwargs.get("sam_imgsz", 640),
                        min_mask_area=kwargs.get("sam_min_area", 80),
                    )

                    # Convert SAM results to DetectionResult and adjust coordinates
                    for r in sam_results:
                        try:
                            from app.ai.ai_result import DetectionResult, PolygonGeometry
                            # Adjust polygon vertices to the original image coordinates
                            verts = [
                                (float(pt[0][0]) + x_min, float(pt[0][1]) + y_min)
                                for pt in r.get("contour", [])
                            ]
                            if len(verts) < 3:
                                continue
                            
                            geometry = PolygonGeometry(vertices=verts)
                            dr = DetectionResult(
                                class_name=r.get("class_name", "roof_polygon"),
                                geometry=geometry,
                                confidence=r["sam_score"],
                                metadata={
                                    "source": f"sam_in_opencv_region_{r.get('class_name', 'unknown')}",
                                    "yolo_confidence": r["score"],
                                    "sam_score": r["sam_score"],
                                    "mask_area": r.get("area_640", 0),
                                },
                            )
                            all_results.append(dr)
                        except Exception as e:
                            logger.warning(f"SAM result conversion failed: {e}")
                logger.info(f"RoofDetector: Refined regions with SAM.")
        
        # The original YOLO and SAM passes are now conditional or removed
        # to avoid duplication. We rely on the new two-step process.
        
        # 3. Roof lines (can still be run on the whole image)
        if kwargs.get("enable_lines", True):
            logger.debug("Running RoofLineDetector...")
            lines = self.line_detector.detect(image)
            all_results.extend(lines)
            logger.info(f"RoofDetector: {len(lines)} lines")

        # 4. Features (disabled by default)
        if kwargs.get("enable_features", False):
            features = self.feature_detector.detect(image)
            all_results.extend(features)

        # 5. Damage (disabled by default)
        if kwargs.get("enable_damage", False):
            damages = self.damage_detector.detect(image)
            all_results.extend(damages)

        logger.info(f"RoofDetector: {len(all_results)} total detections")
        return all_results

    def detect_sam(
        self,
        image: np.ndarray,
        x: Optional[int] = None,
        y: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convenience: SAM-based detection - point-click or auto.

        If (x, y) provided: point-click mode (user clicked a roof point)
        Otherwise: YOLO-assisted auto mode

        Returns dict with 'results' list and 'vis' rendered image.
        """
        if self.sam_segmenter is None:
            raise RuntimeError("SAM not loaded. Call load() with enable_sam=True.")

        if x is not None and y is not None:
            # Point-click mode
            results = self.sam_segmenter.segment_from_point(
                image, x, y,
                multimask=kwargs.get("multimask", False),
            )
            vis = self.sam_segmenter.draw_point_result(
                image, results, (x, y),
                alpha=kwargs.get("alpha", 0.40),
            )
            return {"results": results, "vis": vis, "mode": "point"}
        else:
            # Auto mode
            results = self.sam_segmenter.segment_auto(
                image,
                conf=kwargs.get("conf", 0.25),
                iou=kwargs.get("iou", 0.45),
                imgsz=kwargs.get("imgsz", 640),
                min_mask_area=kwargs.get("min_mask_area", 80),
            )
            vis = self.sam_segmenter.draw_results(
                image, results,
                alpha=kwargs.get("alpha", 0.35),
                show_labels=kwargs.get("show_labels", True),
            )
            return {"results": results, "vis": vis, "mode": "auto"}

    def detect_planes(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Convenience: detect only roof planes."""
        return self.hybrid_detector.detect(image, **kwargs)

    def detect_lines(self, image: np.ndarray, **kwargs) -> List[DetectionResult]:
        """Convenience: detect only roof lines."""
        return self.line_detector.detect(image, **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        return self._model_info

    def validate(self, data: Any) -> bool:
        return isinstance(data, np.ndarray) and data.ndim >= 2 and data.shape[0] > 0 and data.shape[1] > 0

    def unload(self):
        """Unload all models to free memory."""
        if self.sam_segmenter:
            self.sam_segmenter.unload()
            self.sam_segmenter = None
        self._is_loaded = False