"""
YOLO Detection Service

Wrapper around YOLOv8 for object detection and segmentation.
Handles model loading, inference, and result extraction.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger(__name__)


@dataclass
class DetectionBox:
    """Single detection bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class SegmentationMask:
    """Single segmentation mask."""
    mask: np.ndarray  # Binary mask (H, W)
    box: DetectionBox
    area: int  # Number of pixels


@dataclass
class YOLODetectionResult:
    """Result from YOLO inference."""
    boxes: List[DetectionBox]
    masks: List[SegmentationMask]
    image_shape: Tuple[int, int, int]  # (H, W, C)
    inference_time_ms: float
    model_name: str


class YOLODetector:
    """
    YOLOv8 object detector and segmentation model wrapper.
    
    Handles:
    - Model loading with caching
    - Inference on images
    - Mask and box extraction
    - Result conversion to standardized format
    - Error handling and logging
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        cache_model: bool = True
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run on ("auto", "cpu", "cuda", etc.)
            cache_model: Cache model in memory after loading

        Raises:
            ImportError: If ultralytics not installed
            FileNotFoundError: If model file doesn't exist
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.cache_model = cache_model
        self.model: Optional[YOLO] = None
        self._model_load_time = 0.0

    def load(self) -> None:
        """
        Load YOLO model.

        Raises:
            RuntimeError: If model loading fails
        """
        if self.model is not None:
            logger.debug("YOLO model already loaded")
            return

        try:
            logger.info(f"Loading YOLO model from {self.model_path} on device {self.device}...")
            start = time.time()
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self._model_load_time = time.time() - start
            logger.info(f"YOLO model loaded in {self._model_load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"YOLO model loading failed: {e}") from e

    def unload(self) -> None:
        """Unload model and free resources."""
        if self.model is None:
            return

        try:
            logger.info("Unloading YOLO model...")
            del self.model
            self.model = None

            # Clear CUDA cache if available
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

            logger.info("YOLO model unloaded")
        except Exception as e:
            logger.warning(f"Error unloading YOLO model: {e}")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> YOLODetectionResult:
        """
        Run detection on image.

        Args:
            image: Input image (BGR uint8)
            conf_threshold: Override confidence threshold
            iou_threshold: Override IOU threshold

        Returns:
            YOLODetectionResult with boxes and masks

        Raises:
            RuntimeError: If model not loaded or inference fails
            ValueError: If image format invalid
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(f"Invalid image: {type(image)}")

        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold

        try:
            start = time.time()

            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf,
                iou=iou,
                verbose=False
            )

            inference_time = (time.time() - start) * 1000

            if not results or len(results) == 0:
                logger.warning("YOLO returned empty results")
                return YOLODetectionResult(
                    boxes=[],
                    masks=[],
                    image_shape=image.shape,
                    inference_time_ms=inference_time,
                    model_name=Path(self.model_path).stem
                )

            result = results[0]
            boxes, masks = self._extract_detections(result, image)

            logger.debug(
                f"YOLO inference: {len(boxes)} boxes, {len(masks)} masks in {inference_time:.1f}ms"
            )

            return YOLODetectionResult(
                boxes=boxes,
                masks=masks,
                image_shape=image.shape,
                inference_time_ms=inference_time,
                model_name=Path(self.model_path).stem
            )

        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            raise RuntimeError(f"Inference failed: {e}") from e

    def _extract_detections(
        self,
        result,
        image: np.ndarray
    ) -> Tuple[List[DetectionBox], List[SegmentationMask]]:
        """
        Extract boxes and masks from YOLO result.

        Args:
            result: YOLO detection result object
            image: Original image for reference

        Returns:
            Tuple of (boxes, masks)
        """
        boxes: List[DetectionBox] = []
        masks: List[SegmentationMask] = []

        if result.boxes is None or result.boxes.data is None:
            return boxes, masks

        # Extract bounding boxes
        for i, box_data in enumerate(result.boxes.data):
            try:
                x1, y1, x2, y2, conf, cls_id = box_data.cpu().numpy()[:6]
                class_name = result.names.get(int(cls_id), "unknown")

                det_box = DetectionBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=class_name
                )
                boxes.append(det_box)

            except Exception as e:
                logger.warning(f"Failed to extract box {i}: {e}")
                continue

        # Extract segmentation masks
        if result.masks is not None and result.masks.data is not None:
            for i, mask_tensor in enumerate(result.masks.data):
                try:
                    if i >= len(boxes):
                        logger.warning(f"Mask {i} but only {len(boxes)} boxes")
                        break

                    # Convert tensor to numpy
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)

                    # Ensure mask is 2D
                    if mask_np.ndim > 2:
                        mask_np = mask_np.squeeze()

                    # Resize to image size if needed
                    if mask_np.shape != image.shape[:2]:
                        mask_np = cv2.resize(
                            mask_np,
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )

                    # Binarize
                    mask_binary = (mask_np > 0.5).astype(np.uint8)

                    seg_mask = SegmentationMask(
                        mask=mask_binary,
                        box=boxes[i],
                        area=int(mask_binary.sum())
                    )
                    masks.append(seg_mask)

                except Exception as e:
                    logger.warning(f"Failed to extract mask {i}: {e}")
                    continue

        return boxes, masks

    def update_config(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None
    ) -> None:
        """Update detector configuration."""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            logger.info(f"Updated confidence threshold to {conf_threshold}")

        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            logger.info(f"Updated IOU threshold to {iou_threshold}")

        if device is not None:
            self.device = device
            if self.model is not None:
                try:
                    self.model.to(device)
                    logger.info(f"Moved model to device {device}")
                except Exception as e:
                    logger.warning(f"Failed to move model to device: {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def __enter__(self):
        """Context manager support."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.unload()