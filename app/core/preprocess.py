# -*- coding: utf-8 -*-
"""Preprocessing ortofota do „zrozumiteľného" formátu pre ďalšie kroky pipeline.

Čo tento modul robí:
  1. ZBGIS snímku (JPEG z WMS) uloží ako **GeoTIFF** (EPSG:3857) — georeferencovaný
     rastr, takže žiadny ďalší krok nemusí počítať pixely ručne.
  2. Spustí **dlaždicovú inferenciu** natrénovaného segmentačného modelu (640 px
     dlaždice s prevlečením) a masky zošije do jedného rastra → `<name>_masky.tif`.
  3. Vektorizuje masky na **polygóny v S-JTSK (EPSG:8353)** → `<name>_masky.geojson`
     (vstup pre geometrický engine: „kde plocha je"; mračno dodá výšky/sklony).
  4. Zapíše metadáta do `<name>_preprocess.json`.

Prečo dlaždice: model je trénovaný na 640 px; jedno zmenšenie 4096 px obrázka
stratí malé prvky (vikiere, komíny). Prevlеčenie 96 px zabraňuje švom.
"""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# triedy modelu (rovnaké ako v photo pipeline)
CLASS_NAMES = {0: "slope_flat", 1: "slope_min", 2: "slope_poly", 3: "slope_trap", 4: "slope_tri"}


def _geo_tags(bbox, width: int, height: int):
    """Minimálne GeoTIFF tagy pre EPSG:3857 (pixel scale + tiepoint + geo keys)."""
    xmin, ymin, xmax, ymax = bbox
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height
    # tiepoint: raster (0,0) → ľavý horný roh prvého pixelu
    tie = (0.0, 0.0, 0.0, float(xmin), float(ymax), 0.0)
    geokeys = (
        1, 1, 0, 3,            # verzia, revízia, minor, počet kľúčov
        1024, 0, 1, 1,         # GTModelTypeGeoKey = projected
        1025, 0, 1, 1,         # GTRasterTypeGeoKey = PixelIsArea
        3072, 0, 1, 3857,      # ProjectedCSTypeGeoKey = EPSG:3857
    )
    return [
        (33550, "d", 3, (float(sx), float(sy), 0.0)),
        (33922, "d", 6, tie),
        (34735, "H", len(geokeys), geokeys),
    ]


def save_geotiff(img_bgr: np.ndarray, bbox, out_tif: Path) -> Path:
    """Uloží obrázok ako GeoTIFF (EPSG:3857)."""
    import tifffile

    h, w = img_bgr.shape[:2]
    rgb = img_bgr[:, :, ::-1]                      # BGR → RGB
    tifffile.imwrite(str(out_tif), rgb, photometric="rgb",
                     extratags=_geo_tags(bbox, w, h))
    return out_tif


def save_world_file(bbox, width: int, height: int, out_pgw: Path) -> Path:
    """World file (.pgw) — alternatíva k GeoTIFF pre CAD/QGIS."""
    xmin, ymin, xmax, ymax = bbox
    sx = (xmax - xmin) / width
    sy = (ymax - ymin) / height
    out_pgw.write_text("\n".join(f"{v:.6f}" for v in (sx, 0.0, 0.0, -sy,
                                                      xmin + sx / 2, ymax - sy / 2)) + "\n",
                       encoding="ascii")
    return out_pgw


def tiled_masks(img_bgr: np.ndarray, model_path: str, tile: int = 640, overlap: int = 96,
                conf: float = 0.15) -> Tuple[np.ndarray, Dict[str, int]]:
    """Dlaždicová inferencia segmentačného modelu; vráti celkovú masku (triedy) a počty."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    h, w = img_bgr.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    counts: Dict[str, int] = {}
    step = tile - overlap
    xs = list(range(0, max(1, w - overlap), step))
    ys = list(range(0, max(1, h - overlap), step))
    for y0 in ys:
        for x0 in xs:
            x1 = min(w, x0 + tile)
            y1 = min(h, y0 + tile)
            sub = img_bgr[y0:y1, x0:x1]
            if sub.shape[0] < 64 or sub.shape[1] < 64:
                continue
            import cv2
            rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            res = model(rgb, conf=conf, verbose=False)
            if not res or not res[0].masks:
                continue
            masks = res[0].masks.data.cpu().numpy()
            cls = res[0].boxes.cls.cpu().numpy() if res[0].boxes is not None else np.zeros(len(masks))
            mh, mw = sub.shape[:2]
            for m, c in zip(masks, cls):
                name = CLASS_NAMES.get(int(c), f"class_{int(c)}")
                counts[name] = counts.get(name, 0) + 1
                mm = cv2.resize(m, (mw, mh), interpolation=cv2.INTER_LINEAR)
                region = full[y0:y1, x0:x1]
                region[mm > 0.5] = int(c) + 1      # 0 = pozadie
    return full, counts


def masks_to_geojson(mask: np.ndarray, bbox, out_geojson: Path, simplify_px: float = 4.0) -> Path:
    """Vektorizuje masku do GeoJSON (EPSG:8353) — jeden feature na zhluk triedy."""
    import cv2
    from pyproj import Transformer
    from shapely.geometry import Polygon, mapping

    h, w = mask.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    t = Transformer.from_crs("EPSG:3857", "EPSG:8353", always_xy=True)

    def px_to_world(px, py):
        X = xmin + (px / w) * (xmax - xmin)
        Y = ymax - (py / h) * (ymax - ymin)
        x, y = t.transform(X, Y)
        return [float(x), float(y)]

    feats: List[Dict[str, Any]] = []
    for cls_id, name in CLASS_NAMES.items():
        m = (mask == cls_id + 1).astype(np.uint8)
        if int(m.sum()) < 200:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 400:
                continue
            approx = cv2.approxPolyDP(c, simplify_px, True).reshape(-1, 2)
            ring = [px_to_world(px, py) for px, py in approx]
            if len(ring) < 3:
                continue
            poly = Polygon(ring).buffer(0)
            if poly.is_empty or poly.area < 1.0:
                continue
            feats.append({
                "type": "Feature",
                "properties": {"class": name, "area_m2": round(float(poly.area), 2),
                               "source": "vision_model_tiled"},
                "geometry": mapping(poly),
            })

    out_geojson.write_text(json.dumps({"type": "FeatureCollection",
                                       "crs": {"type": "name",
                                               "properties": {"name": "EPSG:8353"}},
                                       "features": feats}, ensure_ascii=False),
                           encoding="utf-8")
    return out_geojson


def run_preprocess(ortho_bytes: bytes, bbox, model_path: str, name: str, out_dir: Path,
                   tile: int = 640, overlap: int = 96) -> Dict[str, Any]:
    """Celý preprocessing: GeoTIFF + world file + dlaždicové masky + GeoJSON."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imdecode(np.frombuffer(ortho_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "obrázok sa nedá dekódovať"}
    h, w = img.shape[:2]

    tif = save_geotiff(img, bbox, out_dir / f"{name}_ortho.tif")
    pgw = save_world_file(bbox, w, h, out_dir / f"{name}_ortho.pgw")
    mask, counts = tiled_masks(img, model_path, tile=tile, overlap=overlap)
    palette = {0: (0, 0, 0), 1: (40, 40, 40), 2: (95, 95, 95), 3: (150, 150, 150),
               4: (205, 205, 205), 5: (245, 245, 245)}
    vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, col in palette.items():
        vis[mask == k] = col
    mask_tif = save_geotiff(vis, bbox, out_dir / f"{name}_masky.tif")
    geojson = masks_to_geojson(mask, bbox, out_dir / f"{name}_masky.geojson")

    meta = {
        "name": name, "size_px": [w, h], "bbox_3857": list(bbox),
        "crs_raster": "EPSG:3857", "crs_vektor": "EPSG:8353",
        "model": Path(model_path).name, "tile": tile, "overlap": overlap,
        "class_counts": counts,
        "outputs": {"ortho_tif": tif.name, "world_file": pgw.name,
                    "masky_tif": mask_tif.name, "masky_geojson": geojson.name},
    }
    (out_dir / f"{name}_preprocess.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
