# -*- coding: utf-8 -*-
"""
generate_dataset.py — Auto-labeling dataset pre YOLO-seg (Fáza 2b).

Pipeline:
    1. Vyber N oblastí v Trnave (grid okolo mesta)
    2. Pre každú oblasť: stiahni ortofoto (ZBGIS WMS, 200x200m)
    3. Nájdi OSM budovy v bboxe (z trnava_buildings.geojson)
    4. Rozsekaj na dlaždice 640px (overlap 15%)
    5. Pre každú dlaždicu: SAM s bbox promptmi (OSM budovy) -> masky
    6. mask_to_yolo_seg -> YOLO .txt anotácie
    7. data.yaml + train/val split

POUŽITIE:
    python tools/generate_dataset.py --areas 12 --tile 640 --out data/datasets/roofs_v1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

# Projekt root
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from app.core.ortho_fetch import fetch_zbgis_ortho  # noqa: E402
from app.ai.dataset import tile_image, mask_to_yolo_seg  # noqa: E402

ZBGIS_EXTENT = 200.0  # m, strana ortofota
ZBGIS_SIZE = 4096


def load_osm_buildings() -> list:
    path = os.path.join(PROJ, "data", "osm", "trnava_buildings.geojson")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


def bbox_of_building(feat) -> tuple:
    coords = feat["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lats), min(lons), max(lats), max(lons)


def area_grid(center_lat, center_lon, areas: int, spacing_m: float = 600.0):
    """Grid oblastí okolo centra (Trnava)."""
    # 1° lat ~ 111.32 km, 1° lon ~ 84.6 km (na 48.4°)
    lat_step = spacing_m / 111320.0
    lon_step = spacing_m / (111320.0 * math.cos(math.radians(48.4)))
    side = int(math.ceil(math.sqrt(areas)))
    out = []
    for i in range(side):
        for j in range(side):
            if len(out) >= areas:
                break
            out.append((center_lat + (i - side // 2) * lat_step,
                        center_lon + (j - side // 2) * lon_step))
    return out


def buildings_in_bbox(buildings, bbox: tuple) -> list:
    """Budovy ktorých stred je v bboxe (s rezervou 20m)."""
    minlat, minlon, maxlat, maxlon = bbox
    # rozšír o ~20m (orto má 200m, oblasť 160m)
    lat_pad = 20 / 111320.0
    lon_pad = 20 / (111320.0 * math.cos(math.radians(48.4)))
    res = []
    for feat in buildings:
        bl, bn, tl, tn = bbox_of_building(feat)
        clat, clon = (bl + tl) / 2, (bn + tn) / 2
        if (minlat - lat_pad <= clat <= maxlat + lat_pad and
                minlon - lon_pad <= clon <= maxlon + lon_pad):
            res.append(feat)
    return res


def geo_to_px(lat, lon, ref_lat, ref_lon, img_size, extent_m):
    """WGS84 -> pixel (lokálna Mercator aproximácia pre malé oblasti)."""
    # Lokálne metre
    mx = (lon - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
    my = (lat - ref_lat) * 111320.0
    scale = img_size / extent_m
    px = (mx + extent_m / 2) * scale
    py = (extent_m / 2 - my) * scale  # Y dole
    return px, py


def main():
    parser = argparse.ArgumentParser(description="Auto-labeling dataset")
    parser.add_argument("--areas", type=int, default=8, help="počet oblastí (grid)")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--out", default="data/datasets/roofs_v1")
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(PROJ, args.out)
    imgs_dir = os.path.join(out_dir, "images", "train")
    lbls_dir = os.path.join(out_dir, "labels", "train")
    val_imgs = os.path.join(out_dir, "images", "val")
    val_lbls = os.path.join(out_dir, "labels", "val")
    for d in [imgs_dir, lbls_dir, val_imgs, val_lbls, os.path.join(out_dir, "cache")]:
        os.makedirs(d, exist_ok=True)

    # SAM model (načítaj raz)
    from segment_anything import sam_model_registry, SamPredictor
    ckpt = os.path.join(PROJ, "ai_models", "sam_vit_b_01ec64.pth")
    print(f"Načítavam SAM ({os.path.getsize(ckpt)//1e6:.0f} MB)...")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to(device="cpu")
    sam.eval()
    predictor = SamPredictor(sam)

    buildings = load_osm_buildings()
    print(f"OSM budov: {len(buildings)}")

    # Grid oblastí okolo Trnavy
    centers = area_grid(48.3774, 17.5882, args.areas, spacing_m=500.0)
    print(f"Oblasti: {len(centers)}")

    tile_id = 0
    n_annotated = 0
    n_empty = 0
    t_start = time.time()

    for ai, (clat, clon) in enumerate(centers):
        # 1. Ortofoto
        ortho_bytes = fetch_zbgis_ortho(clat, clon, extent_m=ZBGIS_EXTENT, size=ZBGIS_SIZE)
        if not ortho_bytes:
            print(f"[{ai}] ortofoto FAIL")
            continue
        img = cv2.imdecode(np.frombuffer(ortho_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(f"[{ai}] ortofoto {clat:.5f},{clon:.5f} OK ({img.shape[1]}px)")

        # 2. Budovy v bboxe
        half = ZBGIS_EXTENT / 2
        bbox = (clat - half / 111320, clon - half / (111320 * math.cos(math.radians(48.4))),
                clat + half / 111320, clon + half / (111320 * math.cos(math.radians(48.4))))
        in_b = buildings_in_bbox(buildings, bbox)
        print(f"    budov v bboxe: {len(in_b)}")

        # 3. Dlaždice
        tiles = tile_image(img, args.tile, args.overlap)
        print(f"    dlaždíc: {len(tiles)}")

        for x0, y0, tile in tiles:
            th, tw = tile.shape[:2]
            # Budovy ktoré pretínajú dlaždicu (bbox v px)
            tile_prompts = []
            for feat in in_b:
                coords = feat["geometry"]["coordinates"][0]
                pxs = [geo_to_px(c[1], c[0], clat, clon, ZBGIS_SIZE, ZBGIS_EXTENT) for c in coords]
                bxs = [p[0] for p in pxs]
                bys = [p[1] for p in pxs]
                bx0, bx1 = min(bxs), max(bxs)
                by0, by1 = min(bys), max(bys)
                # Pretína dlaždicu?
                if bx1 < x0 or bx0 > x0 + tw or by1 < y0 or by0 > y0 + th:
                    continue
                # Orez bbox na dlaždicu
                clip = [max(bx0 - x0, 0), max(by0 - y0, 0),
                        min(bx1 - x0, tw), min(by1 - y0, th)]
                if clip[2] - clip[0] < 20 or clip[3] - clip[1] < 20:
                    continue
                tile_prompts.append(np.array(clip, dtype=float))

            tile_id += 1
            tid = f"t{tile_id:05d}"
            img_path = os.path.join(imgs_dir, f"{tid}.jpg")

            # 4. SAM segmentácia
            mask_sum = np.zeros((th, tw), dtype=np.uint8)
            if tile_prompts:
                predictor.set_image(tile)
                for bp in tile_prompts:
                    try:
                        masks, scores, _ = predictor.predict(box=np.array([bp]), multimask_output=True)
                        best = int(np.argmax(scores))
                        m = masks[best] > 0.5
                        mask_sum[m] = 255
                    except Exception as e:
                        print(f"    SAM chyba: {e}")

            # 5. Ulož obrázok + anotáciu
            cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(img_path)
            if mask_sum.sum() > 500:
                lines = mask_to_yolo_seg(mask_sum, class_id=0)
                if lines:
                    with open(os.path.join(lbls_dir, f"{tid}.txt"), "w") as f:
                        f.write("\n".join(lines) + "\n")
                    n_annotated += 1
                    # Kontrolný overlay
                    if n_annotated <= 5:
                        ov = tile.copy()
                        cnts, _ = cv2.findContours(mask_sum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(ov, cnts, -1, (0, 0, 255), 2)
                        cv2.imencode(".png", ov)[1].tofile(os.path.join(out_dir, "cache", f"chk_{tid}.png"))
                else:
                    n_empty += 1
            else:
                n_empty += 1

            if tile_id % 10 == 0:
                el = time.time() - t_start
                print(f"    ... {tile_id} dlaždíc, {n_annotated} anotovaných, {el:.0f}s")

    # Train/val split (10% val)
    all_ids = [f[:-4] for f in os.listdir(lbls_dir) if f.endswith(".txt")]
    rng = np.random.default_rng(42)
    val_ids = set(rng.choice(all_ids, max(1, int(len(all_ids) * 0.1)), replace=False)) if all_ids else set()
    for tid in all_ids:
        split = "val" if tid in val_ids else "train"
        for ext in (".jpg", ".txt"):
            src = os.path.join(imgs_dir if ext == ".jpg" else lbls_dir, tid + ext)
            dst = os.path.join(val_imgs if ext == ".jpg" else val_lbls, tid + ext)
            if os.path.exists(src) and not os.path.exists(dst):
                os.replace(src, dst)

    # data.yaml
    yaml = (
        f"path: {os.path.abspath(out_dir)}\n"
        f"train: images/train\nval: images/val\nnames:\n  0: roof\n"
    )
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)

    print(f"\n=== HOTOVO ===")
    print(f"Dlaždíc celkom: {tile_id}")
    print(f"Anotovaných: {n_annotated}")
    print(f"Prázdnych: {n_empty}")
    print(f"Čas: {time.time()-t_start:.0f}s")
    print(f"Dataset: {out_dir}")


if __name__ == "__main__":
    main()
