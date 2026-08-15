# -*- coding: utf-8 -*-
"""Konverzia Kaggle rooftops -> YOLO-seg dataset."""
import os, sys, shutil
sys.path.insert(0, r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio")

import numpy as np
import cv2
from app.ai.dataset import mask_to_yolo_seg

DS = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\datasets\kaggle_rooftops"
OUT = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\datasets\roofs_kaggle"

# Štruktúra: images/...  a label/...
img_root = os.path.join(DS, "images", "images")
lbl_root = os.path.join(DS, "label", "label") if os.path.exists(os.path.join(DS, "label", "label")) else os.path.join(DS, "label")
if not os.path.exists(lbl_root):
    # nájdi
    for root, dirs, files in os.walk(os.path.join(DS, "label")):
        if any(f.endswith(".tif") for f in files):
            lbl_root = root
            break
print(f"Images: {img_root}")
print(f"Labels: {lbl_root}")

imgs = sorted(f for f in os.listdir(img_root) if f.endswith(".tif"))
print(f"Obrázkov: {len(imgs)}")

# YOLO dataset štruktúra
for split in ("train", "val"):
    os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

rng = np.random.default_rng(7)
n_val = max(1, int(len(imgs) * 0.15))
val_ids = set(rng.choice(imgs, n_val, replace=False).tolist())

n_ok = 0
n_empty = 0
n_fail = 0
for fname in imgs:
    tid = os.path.splitext(fname)[0]
    split = "val" if fname in val_ids else "train"

    # Obrázok
    img_path = os.path.join(img_root, fname)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        n_fail += 1
        continue

    # Maska: <tid>_label.tif (alebo v lbl_root s rovnakým menom)
    mask_path = os.path.join(lbl_root, f"{tid}_label.tif")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(lbl_root, fname.replace(".tif", "_label.tif"))
    if not os.path.exists(mask_path):
        n_empty += 1
        continue
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        n_fail += 1
        continue
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(np.uint8) * 255

    lines = mask_to_yolo_seg(mask, class_id=0)
    if not lines:
        n_empty += 1
        continue

    # Ulož (konvertuj TIF -> JPG)
    out_img = os.path.join(OUT, "images", split, f"{tid}.jpg")
    cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(out_img)
    with open(os.path.join(OUT, "labels", split, f"{tid}.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    n_ok += 1

# data.yaml
yaml = (
    f"path: {os.path.abspath(OUT)}\n"
    f"train: images/train\nval: images/val\nnames:\n  0: roof\n"
)
with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml)

print(f"\n=== HOTOVO ===")
print(f"Anotovaných: {n_ok}")
print(f"Prázdnych (bez masky/obsahu): {n_empty}")
print(f"Chyby: {n_fail}")
print(f"Train: {len(os.listdir(os.path.join(OUT, 'labels', 'train')))} lbl")
print(f"Val: {len(os.listdir(os.path.join(OUT, 'labels', 'val')))} lbl")
print(f"data.yaml: {os.path.join(OUT, 'data.yaml')}")
