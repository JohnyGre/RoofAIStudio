# -*- coding: utf-8 -*-
"""
generate_cities_v2.py - ZBGIS dlaždice s anotáciami PRIAMO z OSM polygónov.
(BEZ SAM - OSM footprint je presnejší pre vertikálne ortofoto.)

Pre každé mesto: N oblastí -> ZBGIS ortofoto 200x200m -> dlaždice 640px ->
OSM budovy -> rastrová maska (fillPoly) -> YOLO-seg .txt + metadáta JSON.
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
LAT_TRN = 48.4

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


def geo_to_px(lat, lon, ref_lat, ref_lon, img_size, extent_m):
    mx = (lon - ref_lon) * 111320.0 * math.cos(math.radians(ref_lat))
    my = (lat - ref_lat) * 111320.0
    scale = img_size / extent_m
    px = (mx + extent_m / 2) * scale
    py = (extent_m / 2 - my) * scale
    return px, py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--areas", type=int, default=5)
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--out", default="data/datasets/roofs_zbgis")
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.join(PROJ, args.out)
    # Vyčisti starý dataset
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    for d in ["images/train", "labels/train", "images/val", "labels/val", "cache"]:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    rng = random.Random(args.seed)
    tile_id = 0
    n_annotated = 0
    n_empty = 0
    t_start = time.time()
    meta = {}  # dlaždica -> info

    for city in CITIES:
        buildings = load_city_buildings(city)
        if not buildings:
            print(f"{city}: žiadne budovy (preskočené)")
            continue
        print(f"\n=== {city}: {len(buildings)} budov ===", flush=True)

        # Vyber N najväčších budov ako centrá oblastí (paneláky/haly)
        candidates = sorted(buildings, key=lambda f: -len(f["geometry"]["coordinates"][0]))
        chosen = candidates[: max(6, args.areas * 3)]
        rng.shuffle(chosen)
        centers = []
        for feat in chosen[: args.areas * 2]:
            clat, clon = building_center(feat)
            if rng.random() < 0.5:
                centers.append((clat, clon))
            else:
                centers.append((clat + rng.uniform(-0.0012, 0.0012),
                                clon + rng.uniform(-0.0018, 0.0018)))
        centers = centers[: args.areas]

        for ai, (clat, clon) in enumerate(centers):
            ortho_bytes = fetch_zbgis_ortho(clat, clon, extent_m=ZBGIS_EXTENT, size=ZBGIS_SIZE)
            if not ortho_bytes:
                print(f"  [{ai}] ortofoto FAIL", flush=True)
                continue
            img = cv2.imdecode(np.frombuffer(ortho_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            # Budovy v okolí
            half = ZBGIS_EXTENT / 2 + 30
            lat_pad = half / 111320.0
            lon_pad = half / (111320.0 * math.cos(math.radians(LAT_TRN)))
            in_b = []
            for feat in buildings:
                clat2, clon2 = building_center(feat)
                if (clat - lat_pad <= clat2 <= clat + lat_pad and
                        clon - lon_pad <= clon2 <= clon + lon_pad):
                    in_b.append(feat)

            tiles = tile_image(img, args.tile, args.overlap)
            for x0, y0, tile in tiles:
                th, tw = tile.shape[:2]
                mask_sum = np.zeros((th, tw), dtype=np.uint8)
                tile_buildings = []
                for feat in in_b:
                    coords = feat["geometry"]["coordinates"][0]
                    pxs = [geo_to_px(c[1], c[0], clat, clon, ZBGIS_SIZE, ZBGIS_EXTENT) for c in coords]
                    bxs = [p[0] for p in pxs]
                    bys = [p[1] for p in pxs]
                    bx0, bx1 = min(bxs), max(bxs)
                    by0, by1 = min(bys), max(bys)
                    if bx1 < x0 or bx0 > x0 + tw or by1 < y0 or by0 > y0 + th:
                        continue
                    # OSM polygón orezaný do dlaždice -> rastrová maska
                    poly = [(max(min(p[0] - x0, tw - 1), 0), max(min(p[1] - y0, th - 1), 0)) for p in pxs]
                    if len(poly) < 3:
                        continue
                    area_est = abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))) / 2
                    if area_est < 300:  # < ~17x17px
                        continue
                    cv2.fillPoly(mask_sum, [np.array(poly, dtype=np.int32)], 255)
                    tile_buildings.append(feat["properties"].get("osm_id", "?"))

                tile_id += 1
                tid = f"t{tile_id:05d}"
                img_path = os.path.join(out_dir, "images", "train", f"{tid}.jpg")
                cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(img_path)
                meta[tid] = {"city": city, "lat": clat, "lon": clon, "x0": x0, "y0": y0,
                             "buildings": len(tile_buildings), "osm_ids": tile_buildings[:10]}

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

                if tile_id % 30 == 0:
                    el = time.time() - t_start
                    print(f"  ... {tile_id} dlaždíc, {n_annotated} anotovaných, {el:.0f}s", flush=True)

    # Split 10% val (anotované)
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
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"\n=== HOTOVO ===")
    print(f"Dlaždíc celkom: {tile_id}")
    print(f"Anotovaných: {n_annotated}")
    print(f"Prázdnych: {n_empty}")
    print(f"Čas: {time.time()-t_start:.0f}s")
    print(f"Dataset: {out_dir}")


if __name__ == "__main__":
    main()
