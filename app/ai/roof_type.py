# -*- coding: utf-8 -*-
"""
Klasifikátor typu strechy pre pipeline (krok medzi YOLO footprintom a LAZ analýzou).

Používa model roof_gmaps_v2_last.pt (5 tried: slope_flat, slope_min, slope_poly,
slope_trap, slope_tri) na ortofote. Detekcie pretínajúce footprint budovy sa
agregujú vážením konfidenciou -> výsledný typ strechy + distribúcia.

Príklad:
    from app.ai.roof_type import RoofTypeClassifier
    clf = RoofTypeClassifier("ai_models/roof_gmaps_v2_last.pt")
    result = clf.classify(img, footprint_sjtsk, geo_ctx)
    # result = {"roof_type": "slope_poly", "conf": 0.899, "distribution": {...}, "detections": 12}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

logger = logging.getLogger("roofai.roof_type")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Slovenské názvy typov striech (pre výkaz)
ROOF_TYPE_NAMES_SK = {
    "slope_flat": "plochá",
    "slope_min": "minimálny sklon",
    "slope_poly": "polygonálna (valbová)",
    "slope_trap": "lichobežníková",
    "slope_tri": "trojuholníková (štítová)",
}

DEFAULT_MODEL = "ai_models/roof_gmaps_v2_last.pt"


class RoofTypeClassifier:
    """Klasifikuje typ strechy z ortofota pre daný footprint (S-JTSK polygón)."""

    def __init__(self, model_path: str = DEFAULT_MODEL, device: Optional[str] = None,
                 conf: float = 0.25, tile_size: int = 640, tile_step_ratio: float = 0.85):
        if YOLO is None:
            raise ImportError("Ultralytics YOLO nie je nainštalovaný.")
        self._model_path = str(model_path)
        self._model = YOLO(self._model_path)
        self._device = device or ("cuda" if self._model.device.type == "cuda" or _cuda_available() else "cpu")
        self._conf = conf
        self._tile_size = tile_size
        self._step = int(tile_size * tile_step_ratio)
        self._names: Dict[int, str] = self._model.names
        logger.info("RoofTypeClassifier: %s, triedy %s", self._model_path, self._names)

    def detect_all(self, img: np.ndarray) -> List[Tuple[float, int, Tuple[int, int, int, int]]]:
        """Deteguje typy striech na celom ortofote (tiling s prekryvom)."""
        H, W = img.shape[:2]
        dets: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
        for y0 in range(0, H - self._tile_size + 1, self._step):
            for x0 in range(0, W - self._tile_size + 1, self._step):
                tile = img[y0:y0 + self._tile_size, x0:x0 + self._tile_size]
                r = self._model.predict(tile, conf=self._conf, device=self._device, verbose=False)[0]
                for j in range(len(r.boxes)):
                    conf = float(r.boxes.conf[j])
                    cls = int(r.boxes.cls[j])
                    bx0, by0, bx1, by1 = [int(v) for v in r.boxes.xyxy[j].tolist()]
                    dets.append((conf, cls, (x0 + bx0, y0 + by0, x0 + bx1, y0 + by1)))
        return dets

    def classify(self, img: np.ndarray, footprint_px: Optional[Tuple[int, int, int, int]] = None,
                 footprint_poly_px: Optional[np.ndarray] = None,
                 min_conf: float = 0.4) -> Dict[str, Any]:
        """
        Klasifikuje typ strechy.

        Args:
            img: ortofoto (BGR numpy).
            footprint_px: bbox budovy v px (x0, y0, x1, y1); None = celý obrázok.
            footprint_poly_px: presný polygón budovy v px (N×2) — filtruje podľa
                centra detekcie (presnejšie ako bbox).
            min_conf: minimálna conf pre započítanie do distribúcie.

        Returns:
            {"roof_type", "roof_type_sk", "conf", "distribution", "detections",
             "footprint_px"}
        """
        dets = self.detect_all(img)

        # filter podľa bboxu
        if footprint_px is not None:
            fx0, fy0, fx1, fy1 = footprint_px
            hits = []
            for conf, cls, bbox in dets:
                ix0, iy0 = max(bbox[0], fx0), max(bbox[1], fy0)
                ix1, iy1 = min(bbox[2], fx1), min(bbox[3], fy1)
                if max(0, ix1 - ix0) * max(0, iy1 - iy0) > 0:
                    hits.append((conf, cls, bbox))
            dets = hits

        # filter podľa polygónu (centrum detekcie v polygóne)
        if footprint_poly_px is not None:
            from matplotlib.path import Path as MplPath
            pp = MplPath(np.asarray(footprint_poly_px, dtype=float))
            hits = []
            for conf, cls, bbox in dets:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                if pp.contains_point((cx, cy)):
                    hits.append((conf, cls, bbox))
            dets = hits

        # vážená distribúcia podľa conf
        dist: Dict[str, float] = {}
        weight_sum = 0.0
        cls_conf: Dict[int, float] = {}
        for conf, cls, _bbox in dets:
            if conf < min_conf:
                continue
            name = self._names.get(cls, str(cls))
            dist[name] = dist.get(name, 0.0) + conf
            weight_sum += conf
            cls_conf[cls] = max(cls_conf.get(cls, 0.0), conf)

        # primárny typ = najsilnejšia detekcia (max conf); distribúcia = vážený kontext
        if not dets:
            return {"roof_type": "unknown", "roof_type_sk": "neznámy",
                    "conf": 0.0, "distribution": {}, "detections": 0,
                    "footprint_px": footprint_px}

        strong = [d for d in dets if d[0] >= min_conf]
        if not strong:
            strong = dets
        best_det = max(strong, key=lambda d: d[0])
        best_name = self._names.get(best_det[1], str(best_det[1]))
        best_conf = round(float(best_det[0]), 3)
        dist_pct = {k: round(v / weight_sum * 100, 1) for k, v in dist.items()} if weight_sum > 0 else {}

        return {
            "roof_type": best_name,
            "roof_type_sk": ROOF_TYPE_NAMES_SK.get(best_name, best_name),
            "conf": best_conf,
            "distribution": dist_pct,
            "detections": len(dets),
            "footprint_px": footprint_px,
        }


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------- CLI ----------
if __name__ == "__main__":
    import sys, json, pyproj

    PROJ = Path(__file__).resolve().parents[2]
    ORTHO = PROJ / "data" / "ortho" / "Atriova_16H_zbgis_200m.jpg"
    # geo kontext Átrovej
    lat0, lon0 = 48.3961053, 17.587124

    import math

    def to_merc(lat, lon):
        x = lon * 20037508.34 / 180.0
        y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        return x, y * 20037508.34 / 180.0

    cx_m, cy_m = to_merc(lat0, lon0)
    xmin, xmax = cx_m - 100.0, cx_m + 100.0
    ymin, ymax = cy_m - 100.0, cy_m + 100.0
    SCALE = 4096 / 200.0

    t_inv = pyproj.Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
    t_fwd = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)

    def sjtsk_to_px(x, y):
        lon, lat = t_inv.transform(x, y)
        mx, my = to_merc(lat, lon)
        return (mx - xmin) * SCALE, (ymax - my) * SCALE

    img = cv2.imdecode(np.fromfile(str(ORTHO), dtype=np.uint8), cv2.IMREAD_COLOR)

    clf = RoofTypeClassifier(str(PROJ / DEFAULT_MODEL))
    res = clf.classify(img, footprint_px=None)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # s footprintom Átrovej (z finálneho geojsonu)
    fp_file = PROJ / "data" / "exports" / "atriova_16H_footprint_v2.geojson"
    if fp_file.exists():
        import json as _json
        with open(fp_file, encoding="utf-8") as f:
            gj = _json.load(f)
        coords = gj["features"][0]["geometry"]["coordinates"][0][:-1]
        px = np.array([sjtsk_to_px(*t_fwd.transform(lon, lat)) for lon, lat in coords])
        fx0, fy0 = int(px[:, 0].min()), int(px[:, 1].min())
        fx1, fy1 = int(px[:, 0].max()), int(px[:, 1].max())
        res2 = clf.classify(img, footprint_px=(fx0, fy0, fx1, fy1), footprint_poly_px=px)
        print("\nS footprintom Átrovej (bbox + polygón):")
        print(json.dumps(res2, ensure_ascii=False, indent=2))
