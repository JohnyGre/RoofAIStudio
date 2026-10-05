# -*- coding: utf-8 -*-
"""Ortofoto pre adresu — maximálne rozlíšenie + prekrytie geometrie.

Postup:
  1. Zoberieme obal budovy (v EPSG:8353) a pridáme 15 % okraj → najtesnejší výrez.
  2. Stiahneme ZBGIS ortofoto (WMS, 4096 px) pre tento výrez → najvyšší detail
     (typicky 0,5–1,0 cm/px pri budove ~20 m).
  3. Ak ZBGIS neodpovie, skúsime OSM dlaždice (zoom 19) ako fallback.
  4. Nakreslíme roviny (tenké obrysy) a nosné hrany (hrubé, farba podľa typu)
     presne na ortofoto — vznikne kontrolný obrázok „geometria na snímke".
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import requests

WMS_URL = "https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get"
OSM_TILE = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
UA = {"User-Agent": "RoofAIStudio/3.0 (roof orthophoto)"}

EDGE_COLORS_BGR = {
    "h": (255, 163, 77),    # hrebeň
    "n": (159, 211, 56),    # nárožie
    "u": (90, 90, 255),     # úžľabie
    "o": (63, 210, 255),    # odkvap
    "s": (255, 125, 199),   # štít
}


def bbox_3857_from_points(points_8353, margin_frac: float = 0.15) -> Tuple[float, float, float, float]:
    """Obal bodov (EPSG:8353) → EPSG:3857 s okrajom. Vracia (xmin, ymin, xmax, ymax)."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:8353", "EPSG:3857", always_xy=True)
    xs, ys = [], []
    for p in points_8353:
        x, y = t.transform(float(p[0]), float(p[1]))
        xs.append(x); ys.append(y)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    mx = (xmax - xmin) * margin_frac
    my = (ymax - ymin) * margin_frac
    # štvorec, aby snímka nebola deformovaná
    side = max(xmax - xmin + 2 * mx, ymax - ymin + 2 * my)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def fetch_zbgis_bbox(bbox, size: int = 4096, timeout: int = 60) -> Optional[bytes]:
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": "1", "STYLES": "", "CRS": "EPSG:3857",
        "BBOX": ",".join(f"{v:.3f}" for v in bbox),
        "WIDTH": str(size), "HEIGHT": str(size), "FORMAT": "image/jpeg",
    }
    try:
        r = requests.get(WMS_URL, params=params, headers=UA, timeout=timeout, verify=False)
        if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
            return r.content
    except Exception:
        pass
    return None


def fetch_osm_bbox(bbox, zoom: int = 19, timeout: int = 30) -> Optional[bytes]:
    """Fallback: poskladá OSM dlaždice pre daný výrez."""
    from PIL import Image

    def tile_xy(x: float, y: float) -> Tuple[int, int]:
        n = 2 ** zoom
        tx = int((x + 20037508.34) / (2 * 20037508.34) * n)
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / (2 * 20037508.34)))))
        ty = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
        return tx, ty

    xmin, ymin, xmax, ymax = bbox
    tx0, ty1 = tile_xy(xmin, ymin)
    tx1, ty0 = tile_xy(xmax, ymax)
    if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) > 64:
        return None
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            try:
                r = requests.get(OSM_TILE.format(z=zoom, x=tx, y=ty), headers=UA, timeout=timeout)
                if r.status_code == 200:
                    canvas.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), ((tx - tx0) * 256, (ty - ty0) * 256))
            except Exception:
                continue
    buf = io.BytesIO()
    canvas.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def save_overlay(
    img_bytes: bytes,
    bbox,
    planes: List[Dict[str, Any]],
    footprint_xy: Optional[List[List[float]]] = None,
    out_path: str = "",
    min_edge_m: float = 2.5,
) -> str:
    """Nakreslí geometriu (EPSG:8353) na ortofoto a uloží PNG."""
    import cv2
    import numpy as np
    from pyproj import Transformer

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    t = Transformer.from_crs("EPSG:8353", "EPSG:3857", always_xy=True)

    def to_px(x: float, y: float):
        X, Y = t.transform(float(x), float(y))
        return (int(round((X - xmin) / (xmax - xmin) * w)),
                int(round((1.0 - (Y - ymin) / (ymax - ymin)) * h)))

    if footprint_xy:
        pts = np.array([to_px(x, y) for x, y in footprint_xy], dtype=np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)

    for pl in planes:
        v2 = pl.get("vertices_2d") or []
        if len(v2) >= 3:
            pts = np.array([to_px(x, y) for x, y in v2], dtype=np.int32)
            cv2.polylines(img, [pts], True, (140, 140, 140), 1, cv2.LINE_AA)
        for e in pl.get("edges", []):
            if e["type"] == "s" or e["length_m"] < min_edge_m:
                continue
            if not (e.get("exact") or e["type"] == "o"):
                continue
            a, b = to_px(e["start"][0], e["start"][1]), to_px(e["end"][0], e["end"][1])
            thick = 5 if e.get("exact") else 3
            cv2.line(img, a, b, EDGE_COLORS_BGR.get(e["type"], (255, 255, 255)), thick, cv2.LINE_AA)

    # legenda + mierka
    y0 = 26
    for code, txt in (("o", "odkvap"), ("h", "hreben"), ("n", "narozie"), ("u", "uzlabie")):
        cv2.line(img, (16, y0), (58, y0), EDGE_COLORS_BGR[code], 4, cv2.LINE_AA)
        cv2.putText(img, txt, (66, y0 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 30
    cm_px = (xmax - xmin) / w * 100.0
    cv2.putText(img, f"ZBGIS ortofoto {w}x{h} px | {cm_px:.2f} cm/px | EPSG:3857", (16, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, img)
    return out_path
