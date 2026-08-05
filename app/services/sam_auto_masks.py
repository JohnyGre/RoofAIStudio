"""
SAM Automatic Mask Generator — zero-shot roof plane detection without YOLO.

Uses SAM's built-in automatic mask generation (grid of points across the image),
then filters masks by shape, color, and size to keep only roof-like regions.

Much slower than YOLO→SAM, but finds planes that YOLO missed (e.g. low-contrast valleys).
"""
import time, logging
from typing import List, Dict, Any, Optional, Tuple
import cv2, numpy as np

logger = logging.getLogger(__name__)

# Roof-typical HSV ranges (broad, catches most roofing materials)
ROOF_COLOR_RANGES: List[Tuple[np.ndarray, np.ndarray]] = [
    # Reddish roofs (tiles, terracotta)
    (np.array([0, 30, 30]), np.array([25, 255, 255])),
    (np.array([160, 30, 30]), np.array([180, 255, 255])),
    # Dark roofs (slate, asphalt, dark grey)
    (np.array([0, 0, 0]), np.array([180, 40, 120])),
    # Brown/tan roofs
    (np.array([10, 20, 40]), np.array([30, 200, 200])),
    # Blue/green metal roofs
    (np.array([80, 20, 30]), np.array([140, 255, 200])),
    # White/light grey (modern flat roofs)
    (np.array([0, 0, 120]), np.array([180, 30, 255])),
]


def roof_color_score(mask: np.ndarray, image_hsv: np.ndarray) -> float:
    """Score 0-1 how much of the masked region matches roof-typical colors."""
    masked = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    pixels = masked[mask > 0]
    if len(pixels) < 50:
        return 0.0
    total = len(pixels)
    best = 0.0
    for lower, upper in ROOF_COLOR_RANGES:
        in_range = np.all((pixels >= lower) & (pixels <= upper), axis=1)
        score = np.sum(in_range) / total
        best = max(best, score)
    return float(best)


def roof_shape_score(mask: np.ndarray, image_area: int = 0) -> float:
    """Score 0-1 based on shape properties (convex, not too elongated, not too small)."""
    area = np.sum(mask)
    if area < 100:
        return 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)
    if cnt_area < 80:
        return 0.0

    # Relative area filter: skip if mask is too large (>50% image) or too tiny (<0.05%)
    if image_area > 0:
        frac = cnt_area / image_area
        if frac > 0.50 or frac < 0.0005:
            return 0.0

    # Bounding box elongation ratio
    x, y, w, h = cv2.boundingRect(cnt)
    if w == 0 or h == 0:
        return 0.0
    ratio = max(w, h) / min(w, h)
    if ratio > 10:  # too elongated (road, edge)
        return 0.0

    # Convexity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return 0.0
    solidity = cnt_area / hull_area
    if solidity < 0.25:  # very concave (tree, bush)
        return 0.0

    # Solidity bonus
    box_area = w * h
    fill_ratio = cnt_area / max(box_area, 1)
    return min(solidity * fill_ratio, 1.0)


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union > 0 else 0.0


def deduplicate_masks(masks: List[Dict], iou_threshold: float = 0.60) -> List[Dict]:
    """Remove duplicate/overlapping masks, keeping the higher-scored one."""
    if len(masks) <= 1:
        return masks
    sorted_masks = sorted(masks, key=lambda m: m["score"], reverse=True)
    keep = []
    for m in sorted_masks:
        overlap = False
        for k in keep:
            if iou(m["segmentation"], k["segmentation"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(m)
    return keep


def generate_roof_masks(
    image: np.ndarray,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.82,
    stability_score_thresh: float = 0.85,
    min_area_frac: float = 0.0005,  # minimum 0.05% of image area (~200px)
    max_area_frac: float = 0.50,    # maximum 50% of image area
    color_weight: float = 0.3,
    shape_weight: float = 0.7,
    dedup_iou: float = 0.50,
    max_output: int = 25,           # cap on returned masks
) -> List[Dict[str, Any]]:
    """
    Generate roof candidate masks using SAM's automatic mode.

    Args:
        image: BGR image (640x640 recommended)
        points_per_side: SAM grid density (higher = more masks found, slower)
        pred_iou_thresh: SAM IoU prediction threshold
        stability_score_thresh: SAM stability threshold
        min_area_frac: Minimum mask area as fraction of image
        max_area_frac: Maximum mask area as fraction of image
        color_weight: Weight of color score in composite score
        shape_weight: Weight of shape score in composite score
        dedup_iou: IoU threshold for deduplication
        max_output: Hard cap on number of returned masks

    Returns:
        List of dicts, each with: mask, contour, score, area_px, color_score, shape_score
    """
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("segment-anything not installed")

    # Load SAM
    sam = sam_model_registry["vit_b"](checkpoint="ai_models/sam_vit_b_01ec64.pth")
    sam.to("cpu")
    sam.eval()

    h, w = image.shape[:2]
    image_area = h * w
    min_mask_area = int(image_area * min_area_frac)
    max_mask_area = int(image_area * max_area_frac)
    t0 = time.time()

    # Generate all masks
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_area,
    )
    raw_masks = mask_generator.generate(image)
    raw_time = time.time() - t0
    logger.info("SAM auto: %d raw masks in %.1fs (image %dx%d, area range %d-%dpx)",
                len(raw_masks), raw_time, w, h, min_mask_area, max_mask_area)

    # Convert to HSV for color scoring
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Filter and score masks
    scored = []
    for m in raw_masks:
        mask = m["segmentation"].astype(np.uint8)
        area = int(m["area"])

        # Relative area filter
        if area < min_mask_area or area > max_mask_area:
            continue

        # Scores
        cs = roof_color_score(mask, image_hsv)
        ss = roof_shape_score(mask, image_area=image_area)
        composite = cs * color_weight + ss * shape_weight

        if composite < 0.10:
            continue

        # Polygon from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        best = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(best, True)
        approx = cv2.approxPolyDP(best, epsilon, True)

        scored.append({
            "mask": mask.astype(bool),
            "segmentation": m["segmentation"],
            "contour": approx,
            "score": composite,
            "area_px": area,
            "color_score": cs,
            "shape_score": ss,
            "sam_iou": float(m.get("predicted_iou", 0)),
            "sam_stability": float(m.get("stability_score", 0)),
        })

    # Deduplicate
    filtered = deduplicate_masks(scored, dedup_iou)
    filtered = sorted(filtered, key=lambda m: m["score"], reverse=True)

    # Hard cap — never return more than max_output masks
    filtered = filtered[:max_output]

    total_time = time.time() - t0
    logger.info(
        "SAM auto filtered: %d masks (from %d raw, composite>=0.25) in %.1fs",
        len(filtered), len(raw_masks), total_time,
    )

    return filtered
