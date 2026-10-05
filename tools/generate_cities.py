# -*- coding: utf-8 -*-
"""
generate_cities.py - Auto-labeling dlaždice pre viac slovenských miest (SAM na GPU).

Pre každé mesto: grid oblastí -> ZBGIS ortofoto 200x200m -> dlaždice 640px ->
OSM budovy (bbox prompt) -> SAM masky -> YOLO-seg anotácie -> train/val split.

POUŽITIE:
    python tools/generate_cities.py --areas 4 --out data/datasets/roofs_zbgis
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from app.core.ortho_fetch import fetch_zbgis_ortho  # noqa: E402
from app.ai.dataset import tile_image, mask_to_yolo_seg  # noqa: E402

ZBGIS_EXTENT = 200.0
ZBGIS_SIZE = 4096

CITIES = {
    "bratislava": "Bratislava, Slovensko",
    "nitra": "Nitra, Slovensko",
    "zilina": "Žilina, Slovensko",
    "kosice": "Košice, Slovensko",
    "presov": "Prešov, Slovensko",
    "banska_bystrica": "Banská Bystrica, Slovensko",
    "trencin": "Trenčín, Slovensko",
    "poprad": "Poprad, Slovensko",
}

LAT_TRN = 48.4


def load_city_buildings(city: str) -> list:
    path = os.path.join(PROJ, "data", "osm", f"{city}_buildings.geojson")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)["features"]


def building_center(feat) -> tuple:
    coords = feat["geometry"]["coordinates"][0]
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def building_bbox_deg(feat) -> tuple:
    coords = feat["geometry"]["coordinates"][0]
    lats = [c[1] for c in coords]
    lons = [c[0] for c in coords]
    return min(lats), min(lons), max(lats), max(lons)


def geo_to_px(lat, lon, ref_lat, ref_lon, img_size, extent_m):
    mx = (lon - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
    my = (lat - ref_lat) * 111320.0
    scale = img_size / extent_m
    px = (mx + extent_m / 2) * scale
    py = (extent_m / 2 - my) * scale
    return px, py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--areas", type=int, default=4, help="oblastí na mesto")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--out", default="data/datasets/roofs_zbgis")
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(PROJ, args.out)
    for d in ["images/train", "labels/train", "images/val", "labels/val", "cache"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # SAM na GPU
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = os.path.join(PROJ, "ai_models", "sam_vit_b_01ec64.pth")
    print(f"SAM na {device}...", flush=True)
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    rng = random.Random(args.seed)
    tile_id = 0
    n_annotated = 0
    n_empty = 0
    t_start = time.time()

    for city in CITIES:
        buildings = load_city_buildings(city)
        if not buildings:
            print(f"{city}: žiadne budovy (preskočené)")
            continue
        print(f"\n=== {city}: {len(buildings)} budov ===", flush=True)

        # Vyber oblasti: N náhodných budov (rôzne veľkosti preferované)
        candidates = sorted(buildings, key=lambda f: -len(f["geometry"]["coordinates"][0]))
        chosen = candidates[: max(5, args.areas * 3)]
        rng.shuffle(chosen)
        centers = []
        for feat in chosen[: args.areas * 2]:
            # niektoré oblasti posunieme od stredu budovy (zachytí okolie)
            clat, clon = building_center(feat)
            if rng.random() < 0.5:
                centers.append((clat, clon))
            else:
                dlat = rng.uniform(-0.0012, 0.0012)
                dlon = rng.uniform(-0.0018, 0.0018)
                centers.append((clat + dlat, clon + dlon))
        centers = centers[: args.areas]

        for ai, (clat, clon) in enumerate(centers):
            ortho_bytes = fetch_zbgis_ortho(clat, clon, extent_m=ZBGIS_EXTENT, size=ZBGIS_SIZE)
            if not ortho_bytes:
                print(f"  [{ai}] ortofoto FAIL", flush=True)
                continue
            img = cv2.imdecode(np.frombuffer(ortho_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Budovy v okolí (stred v bboxe +30m)
            half = ZBGIS_EXTENT / 2 + 30
            bbox = (clat - half / 111320, clon - half / (111320 * math.cos(math.radians(LAT_TRN))),
                    clat + half / 111320, clon + half / (111320 * math.cos(math.radians(LAT_TRN))))
            in_b = []
            for feat in buildings:
                bl, bn, tl, tn = building_bbox_deg(feat)
                cc = ((bl + tl) / 2, (bn + tn) / 2)
                if (bbox[0] <= cc[0] <= bbox[2] and bbox[1] <= cc[1] <= bbox[3]):
                    in_b.append(feat)

            tiles = tile_image(img, args.tile, args.overlap)
            for x0, y0, tile in tiles:
                th, tw = tile.shape[:2]
                tile_prompts = []
                for feat in in_b:
                    coords = feat["geometry"]["coordinates"][0]
                    pxs = [geo_to_px(c[1], c[0], clat, clon, ZBGIS_SIZE, ZBGIS_EXTENT) for c in coords]
                    bxs = [p[0] for p in pxs]
                    bys = [p[1] for p in pxs]
                    bx0, bx1 = min(bxs), max(bxs)
                    by0, by1 = min(bys), max(bys)
                    if bx1 < x0 or bx0 > x0 + tw or by1 < y0 or by0 > y0 + th:
                        continue
                    clip = [max(bx0 - x0, 0), max(by0 - y0, 0),
                            min(bx1 - x0, tw), min(by1 - y0, th)]
                    if clip[2] - clip[0] < 20 or clip[3] - clip[1] < 20:
                        continue
                    tile_prompts.append(np.array(clip, dtype=float))

                tile_id += 1
                tid = f"t{tile_id:05d}"

                mask_sum = np.zeros((th, tw), dtype=np.uint8)
                if tile_prompts:
                    try:
                        predictor.set_image(tile)
                        for bp in tile_prompts:
                            masks, scores, _ = predictor.predict(
                                box=np.array([bp]), multimask_output=True)
                            best = int(np.argmax(scores))
                            m = masks[best] > 0.5
                            mask_sum[m] = 255
                    except Exception as e:
                        print(f"    SAM chyba: {e}", flush=True)

                img_path = os.path.join(out_dir, "images", "train", f"{tid}.jpg")
                cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(img_path)
                if mask_sum.sum() > 500:
                    lines = mask_to_yolo_seg(mask_sum, class_id=0)
                    if lines:
                        with open(os.path.join(out_dir, "labels", "train", f"{tid}.txt"), "w", encoding="utf-8") as f:
                            f.write("\n".join(lines) + "\n")
                        n_annotated += 1
                    else:
                        n_empty += 1
                else:
                    n_empty += 1

                if tile_id % 20 == 0:
                    el = time.time() - t_start
                    print(f"  ... {tile_id} dlaždíc, {n_annotated} anotovaných, {el:.0f}s", flush=True)

    # Split 10% val
    all_ids = [f[:-4] for f in os.listdir(os.path.join(out_dir, "labels", "train")) if f.endswith(".txt")]
    rng2 = np.random.default_rng(42)
    val_ids = set(rng2.choice(all_ids, max(1, int(len(all_ids) * 0.1)), replace=False)) if all_ids else set()
    for tid in all_ids:
        split = "val" if tid in val_ids else "train"
        for ext in (".jpg", ".txt"):
            src = os.path.join(out_dir, "images" if ext == ".jpg" else "labels", "train", tid + ext)
            dst = os.path.join(out_dir, "images" if ext == ".jpg" else "labels", "val", tid + ext)
            if os.path.exists(src) and not os.path.exists(dst):
                os.replace(src, dst)

    yaml = f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\nnames:\n  0: roof\n"
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
