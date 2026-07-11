"""
This module provides an adapter for Ultralytics YOLO segmentation models,
implementing the AbstractSegmentationModel interface.
"""

import uuid
from typing import Any, Dict, List, Optional, Union
import numpy as np
import cv2
import torch # For CUDA detection

from app.ai.segmentation_model import AbstractSegmentationModel
from app.ai.segmentation_result import SegmentationResult
from app.core.logger import setup_logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    setup_logging().warning("Ultralytics YOLO not installed. YOLOSegmentationModel will not be functional.")

logger = setup_logging()

class YOLOSegmentationModel(AbstractSegmentationModel):
    """
    Adapter for Ultralytics YOLO segmentation models.
    It loads a YOLO model, performs segmentation, and returns structured SegmentationResult objects.
    """
    DEFAULT_MODEL_NAME = "YOLO_Segmentation_Adapter"
    DEFAULT_VERSION = "8.x"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, version: str = DEFAULT_VERSION, model_id: Optional[uuid.UUID] = None):
        super().__init__(model_name, version, model_id)
        self._yolo_model: Optional[YOLO] = None
        self._model_path: Optional[str] = None
        self._device: str = "cpu"
        self._class_names: List[str] = []

    def load(self, model_path: str, device: Optional[str] = None, **kwargs) -> None:
        """
        Loads the YOLO segmentation model from the specified path.

        Args:
            model_path (str): Path to the YOLO model weights (e.g., 'yolov8n-seg.pt').
            device (Optional[str]): Device to run the model on ('cpu', 'cuda', '0', '1', etc.).
                                    If None, attempts to use CUDA if available, otherwise CPU.
            **kwargs: Additional keyword arguments for YOLO model loading (e.g., 'task').
        """
        if YOLO is None:
            raise ImportError("Ultralytics YOLO is not installed. Cannot load YOLOSegmentationModel.")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model weights not found at: {model_path}")

        self._model_path = model_path
        
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device for YOLO model: {self._device}")
        else:
            self._device = device
            if "cuda" in self._device and not torch.cuda.is_available():
                logger.warning(f"CUDA device '{self._device}' requested, but CUDA is not available. Falling back to CPU.")
                self._device = "cpu"

        try:
            self._yolo_model = YOLO(self._model_path, **kwargs)
            self._is_loaded = True
            # Get class names if available
            if hasattr(self._yolo_model, 'names'):
                self._class_names = [self._yolo_model.names[i] for i in sorted(self._yolo_model.names.keys())]
            logger.info(f"YOLO segmentation model '{self.model_name}' loaded from {model_path} on device {self._device}.")
        except Exception as e:
            self._is_loaded = False
            self._yolo_model = None
            logger.error(f"Failed to load YOLO model from {model_path}: {e}")
            raise

    def segment(self, image: np.ndarray, confidence_threshold: float = 0.5, iou_threshold: float = 0.7, **kwargs) -> List[SegmentationResult]:
        """
        Performs segmentation on the input image using the loaded YOLO model.

        Args:
            image (np.ndarray): The input image data (NumPy array, BGR format).
            confidence_threshold (float): Minimum confidence score to keep a detection.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression.
            **kwargs: Additional keyword arguments for YOLO predict method (e.g., 'imgsz').

        Returns:
            List[SegmentationResult]: A list of structured SegmentationResult objects.
        """
        if not self.is_loaded or self._yolo_model is None:
            raise RuntimeError(f"YOLO model '{self.model_name}' is not loaded. Call load() first.")
        if not self.validate(image):
            raise ValueError("Invalid image input for segmentation.")

        results: List[SegmentationResult] = []
        original_h, original_w = image.shape[:2]

        try:
            # YOLO expects RGB image, convert from BGR (OpenCV default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform inference
            yolo_predictions = self._yolo_model.predict(
                source=image_rgb,
                conf=confidence_threshold,
                iou=iou_threshold,
                device=self._device,
                verbose=False, # Suppress verbose output
                **kwargs
            )

            for pred in yolo_predictions:
                if pred.masks is not None and pred.boxes is not None:
                    for i, mask_tensor in enumerate(pred.masks.data):
                        box = pred.boxes[i]
                        confidence = box.conf.item()
                        class_id = int(box.cls.item())
                        class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"Class_{class_id}"

                        # Convert mask tensor to numpy array and resize to original image dimensions
                        mask_np = mask_tensor.cpu().numpy()
                        # Resize mask to original image size
                        mask_resized = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                        # Binarize the mask (values > 0.5 become 1, others 0)
                        binary_mask = (mask_resized > 0.5).astype(np.uint8)

                        results.append(SegmentationResult(
                            mask=binary_mask,
                            class_name=class_name,
                            confidence=confidence,
                            image_size=(original_w, original_h),
                            metadata={"model_name": self.model_name, "model_version": self.version}
                        ))
        except Exception as e:
            logger.error(f"Error during YOLO segmentation inference: {e}")
            raise

        logger.info(f"YOLO segmentation completed. Found {len(results)} masks.")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns detailed information about the loaded YOLO model.
        """
        info = {
            "name": self.model_name,
            "version": self.version,
            "model_path": self._model_path,
            "device": self._device,
            "loaded": self.is_loaded,
            "class_names": self._class_names,
            "description": "Ultralytics YOLO segmentation model adapter."
        }
        return info

    def unload(self) -> None:
        """
        Unloads the YOLO model from memory.
        """
        del self._yolo_model
        self._yolo_model = None
        self._is_loaded = False
        logger.info(f"YOLO model '{self.model_name}' unloaded.")
        # Attempt to clear CUDA cache if applicable
        if "cuda" in self._device and torch.cuda.is_available():
            torch.cuda.empty_cache()
