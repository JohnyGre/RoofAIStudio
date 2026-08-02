"""
SAHI-based tiled detector for large orthophoto/satellite/drone images.
Slices large images into overlapping tiles, runs YOLO on each tile,
then merges results with NMS deduplication across tile boundaries.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass

from app.core.logger import setup_logging

logger = setup_logging()


@dataclass
class TilePrediction:
    """A single prediction from one tile."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in original image coords
    mask: Optional[np.ndarray]  # Binary mask if segmentation model
    confidence: float
    class_id: int
    class_name: str
    tile_id: int


class SAHIDetector:
    """
    Slicing Aided Hyper Inference — tiles large images for YOLO processing.
    
    Handles:
    - Automatic tile sizing based on model input size
    - Overlapping tiles to avoid boundary artifacts
    - Post-processing NMS across tile boundaries
    - Mask stitching for segmentation
    """

    def __init__(
        self,
        model,
        tile_size: int = 640,
        overlap_ratio: float = 0.2,
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ):
        """
        Args:
            model: Loaded YOLO model (from ultralytics)
            tile_size: Size of each tile (square), should match model input
            overlap_ratio: Overlap between adjacent tiles (0.0-0.5)
            conf_threshold: Confidence threshold for YOLO predictions
            iou_threshold: IoU threshold for NMS deduplication
        """
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap_ratio
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def _compute_tiles(
        self, img_w: int, img_h: int
    ) -> List[Tuple[int, int, int, int]]:
        """Compute tile coordinates (x1, y1, x2, y2) for the image."""
        tiles = []
        stride = int(self.tile_size * (1 - self.overlap))

        for y in range(0, img_h, stride):
            for x in range(0, img_w, stride):
                x2 = min(x + self.tile_size, img_w)
                y2 = min(y + self.tile_size, img_h)
                x1 = max(0, x2 - self.tile_size)
                y1 = max(0, y2 - self.tile_size)
                tiles.append((x1, y1, x2, y2))

                # Stop if we've reached the edge
                if x2 >= img_w and y2 >= img_h:
                    break
            if y2 >= img_h:
                break

        # Ensure we don't have too many tiles (sanity check)
        if len(tiles) > 200:
            logger.warning(
                f"SAHI: {len(tiles)} tiles — very large image. "
                "Consider increasing tile_size or decreasing overlap."
            )

        return tiles

    def _process_tile(
        self,
        image: np.ndarray,
        tile_coords: Tuple[int, int, int, int],
        tile_id: int,
    ) -> List[TilePrediction]:
        """Run YOLO on a single tile and map results back to original coords."""
        x1, y1, x2, y2 = tile_coords
        tile = image[y1:y2, x1:x2]
        h_tile, w_tile = tile.shape[:2]

        predictions: List[TilePrediction] = []

        try:
            results = self.model.predict(
                tile,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"SAHI tile {tile_id} failed: {e}")
            return predictions

        for r in results:
            if r.boxes is None:
                continue

            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls.item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf.item())
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                # Map tile coords back to original image coords
                global_x1 = int(bx1) + x1
                global_y1 = int(by1) + y1
                global_x2 = int(bx2) + x1
                global_y2 = int(by2) + y1

                # Extract mask if segmentation model
                mask = None
                if r.masks is not None and i < len(r.masks.data):
                    mask_tile = r.masks.data[i].cpu().numpy()
                    mask_tile = (mask_tile > 0.5).astype(np.uint8)
                    # Resize mask to tile size
                    mask_resized = cv2.resize(
                        mask_tile, (w_tile, h_tile),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    # Create full-image-sized mask and place this tile's mask
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    mask[y1:y2, x1:x2] = mask_resized

                predictions.append(TilePrediction(
                    bbox=(global_x1, global_y1, global_x2, global_y2),
                    mask=mask,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    tile_id=tile_id,
                ))

        return predictions

    @staticmethod
    def _nms_deduplicate(
        predictions: List[TilePrediction],
        iou_threshold: float = 0.5,
    ) -> List[TilePrediction]:
        """Non-maximum suppression across tile boundaries."""
        if not predictions:
            return []

        # Sort by confidence descending
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        kept: List[TilePrediction] = []

        for p in predictions:
            x1, y1, x2, y2 = p.bbox
            is_duplicate = False

            for k in kept:
                kx1, ky1, kx2, ky2 = k.bbox
                ix = max(x1, kx1)
                iy = max(y1, ky1)
                iw = min(x2, kx2) - ix
                ih = min(y2, ky2) - iy

                if iw <= 0 or ih <= 0:
                    continue

                inter = iw * ih
                area_p = (x2 - x1) * (y2 - y1)
                area_k = (kx2 - kx1) * (ky2 - ky1)
                union = area_p + area_k - inter
                iou = inter / union if union > 0 else 0

                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(p)

        return kept

    def detect(
        self,
        image: np.ndarray,
        **kwargs,
    ) -> List[TilePrediction]:
        """
        Run tiled detection on a large image.

        Args:
            image: BGR numpy array
            **kwargs: Override any init parameter

        Returns:
            List[TilePrediction] with coordinates in original image space
        """
        tile_size = kwargs.get("tile_size", self.tile_size)
        overlap = kwargs.get("overlap_ratio", self.overlap)
        conf_threshold = kwargs.get("conf_threshold", self.conf_threshold)
        iou_threshold = kwargs.get("iou_threshold", self.iou_threshold)

        h, w = image.shape[:2]

        # If image is smaller than tile_size, run directly
        if w <= tile_size and h <= tile_size:
            return self._process_tile(image, (0, 0, w, h), 0)

        tiles = self._compute_tiles(w, h)
        logger.info(
            f"SAHI: {w}x{h} image -> {len(tiles)} tiles "
            f"({tile_size}px, {overlap*100:.0f}% overlap)"
        )

        all_preds: List[TilePrediction] = []
        for i, tile_coords in enumerate(tiles):
            preds = self._process_tile(image, tile_coords, i)
            all_preds.extend(preds)

        # NMS across tiles
        merged = self._nms_deduplicate(all_preds, iou_threshold)
        logger.info(
            f"SAHI: {len(all_preds)} raw -> {len(merged)} after NMS"
        )

        return merged
