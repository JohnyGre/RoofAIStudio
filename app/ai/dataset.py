# -*- coding: utf-8 -*-
"""
dataset.py — Dataset pipeline pre tréning YOLO-seg (Fáza 2b).

Moduly:
    1. tiling: ortofoto -> dlaždice (640/1024 px + overlap) s geotransformáciou
    2. mask_to_yolo_seg: binárna maska -> YOLO .txt anotácia (normalizovaná)
    3. polygons_to_mask: OSM/kataster polygóny (S-JTSK) -> pixelová maska
    4. build_dataset: kompletná generácia datasetu (images + labels)

Príklad:
    from app.ai.dataset import tile_image, mask_to_yolo_seg
"""
from __future__ import annotations

import json
import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- tiling ---

def tile_image(
    img: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.15,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Rozseká obrázok na dlaždice s presahom.

    Returns:
        [(x0, y0, tile), ...] — pozícia v pôvodnom obrázku + dlaždica
    """
    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap))
    tiles = []
    y0 = 0
    while y0 < h:
        x0 = 0
        while x0 < w:
            tile = img[y0:y0 + tile_size, x0:x0 + tile_size]
            # Doplniť okraje (reflekt) ak dlaždica presahuje
            th, tw = tile.shape[:2]
            if th < tile_size or tw < tile_size:
                pad_b = max(0, tile_size - th)
                pad_r = max(0, tile_size - tw)
                tile = cv2.copyMakeBorder(tile, 0, pad_b, 0, pad_r,
                                          cv2.BORDER_REFLECT_101)
            tiles.append((x0, y0, tile))
            x0 += step
            if x0 + tile_size >= w and x0 + tile_size != w:
                x0 = max(0, w - tile_size)
                break
        y0 += step
        if y0 + tile_size >= h and y0 + tile_size != h:
            y0 = max(0, h - tile_size)
            break
    return tiles


# ------------------------------------------------------- mask -> yolo seg ---

def mask_to_yolo_seg(
    mask: np.ndarray,
    class_id: int = 0,
    min_area: int = 50,
    epsilon_ratio: float = 0.005,
) -> List[str]:
    """
    Binárna maska (0/255) -> YOLO segmentation riadky (normalizované 0-1).

    Args:
        mask: binárna maska (H, W), pozadie 0, objekt 255
        class_id: YOLO class id
        min_area: minimálna plocha kontúry (px) — filter šumu
        epsilon_ratio: DP zjednodušenie (0.005 = 0.5% obvodu)

    Returns:
        [f"{class_id} x1 y1 x2 y2 ...", ...]
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        coords = []
        for point in approx:
            x, y = point[0]
            coords.extend([round(x / w, 6), round(y / h, 6)])

        if len(coords) >= 6:  # min 3 body
            lines.append(f"{class_id} " + " ".join(map(str, coords)))

    return lines


# --------------------------------------------------- polygons -> mask ----

def polygons_to_mask(
    polygons_xy: List[np.ndarray],
    img_w: int,
    img_h: int,
    affine: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Polygóny (v geografických súradniciach) -> binárna maska v pixeloch.

    Args:
        polygons_xy: list polygónov (N, 2) v geografickom CRS
        img_w, img_h: rozmery obrázka
        affine: 3x3 transformácia geo->pixel; ak None, polygóny sú už v px
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in polygons_xy:
        pts = poly
        if affine is not None:
            # geo -> pixel: p_px = affine @ p_geo
            ones = np.column_stack([pts, np.ones(len(pts))])
            pts = (affine @ ones.T).T[:, :2]
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def make_affine_from_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int,
) -> np.ndarray:
    """
    Affine transformácia z geografického bboxu do pixelov.
    Predpoklad: osi sú rovnobežné (WMS/WMTS obrázky).

    pixel_x = (geo_x - xmin) * img_w / (xmax - xmin)
    pixel_y = (ymax - geo_y) * img_h / (ymax - ymin)   # Y otočený
    """
    sx = img_w / (xmax - xmin)
    sy = img_h / (ymax - ymin)
    return np.array([
        [sx, 0, -xmin * sx],
        [0, -sy, ymax * sy],
        [0, 0, 1],
    ], dtype=float)


# ------------------------------------------------------ dataset builder ---

def build_yolo_dataset(
    ortho_tiles_dir: str,
    masks_dir: str,
    out_dir: str,
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Z dlaždíc + masiek vygeneruje YOLO-seg dataset štruktúru:
        out_dir/
          images/train/*.jpg, images/val/*.jpg
          labels/train/*.txt, labels/val/*.txt
          data.yaml

    Očakáva pre každú dlaždicu `<id>.jpg` masku `<id>.png` (binárnu).
    """
    os.makedirs(os.path.join(out_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "val"), exist_ok=True)

    tiles = sorted(f for f in os.listdir(ortho_tiles_dir) if f.endswith(".jpg"))
    if not tiles:
        raise FileNotFoundError(f"Žiadne dlaždice v {ortho_tiles_dir}")

    # 80/20 split
    n_val = max(1, int(len(tiles) * 0.2))
    val_ids = set(np.random.choice(tiles, n_val, replace=False))

    n_empty = 0
    n_annotated = 0
    for fname in tiles:
        tile_id = os.path.splitext(fname)[0]
        split = "val" if fname in val_ids else "train"

        # Skopíruj obrázok
        src_img = os.path.join(ortho_tiles_dir, fname)
        dst_img = os.path.join(out_dir, "images", split, fname)
        if not os.path.exists(dst_img):
            os.replace(src_img, dst_img) if False else _copyfile(src_img, dst_img)

        # Maska
        mask_path = os.path.join(masks_dir, f"{tile_id}.png")
        if not os.path.exists(mask_path):
            n_empty += 1
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.sum() == 0:
            n_empty += 1
            continue

        lines = mask_to_yolo_seg(mask, class_id=0)
        if not lines:
            n_empty += 1
            continue

        with open(os.path.join(out_dir, "labels", split, f"{tile_id}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        n_annotated += 1

    # data.yaml
    names = class_names or ["roof"]
    yaml = (
        f"path: {os.path.abspath(out_dir)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))
    )
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    return {
        "tiles": len(tiles),
        "annotated": n_annotated,
        "empty": n_empty,
        "data_yaml": os.path.join(out_dir, "data.yaml"),
    }


def _copyfile(src: str, dst: str) -> None:
    import shutil
    shutil.copy2(src, dst)
