# -*- coding: utf-8 -*-
"""Vision zdroj: vycvičený segmentačný model → obrys strechy v S-JTSK.

Zapojenie do architektúry (podľa rešerše):
  * Point Cloud = autorita Z / sklonu / plochy,
  * GIS = autorita pôdorysu (keď je dostupná geometria),
  * **Vision = planimetrický podklad a kontrola** — keď GIS geometriu nemá,
    použije sa obrys z modelu (s príznakom v kontrakte, nikdy potichu).

Model: `ai_models/roof_gmaps_v2_last.pt` (5 tried: slope_flat/min/poly/trap/tri).
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PAD = 114
IMGSZ = 640

CLASS_NAMES = {0: "slope_flat", 1: "slope_min", 2: "slope_poly", 3: "slope_trap", 4: "slope_tri"}


def _letterbox(img: np.ndarray, size: int = IMGSZ):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    import cv2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    padded = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                                cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return padded, scale, top, left


def detect_from_ortho(
    ortho_bytes: bytes,
    bbox3857: Tuple[float, float, float, float],
    model_path: str,
    conf: float = 0.15,
    iou: float = 0.7,
) -> Dict[str, Any]:
    """Spustí segmentáciu na ortofote a vráti obrys strechy v EPSG:8353."""
    import cv2
    from pyproj import Transformer
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    arr = np.frombuffer(ortho_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "ortofoto sa nedá dekódovať"}
    H, W = img.shape[:2]

    proc, scale, top, left = _letterbox(img, IMGSZ)

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    res = model(rgb, conf=conf, iou=iou, verbose=False)

    masks: List[np.ndarray] = []
    mask_classes: List[str] = []
    classes: Dict[str, int] = {}
    if res and len(res) > 0 and res[0].masks is not None:
        data = res[0].masks.data.cpu().numpy()
        cls = res[0].boxes.cls.cpu().numpy() if res[0].boxes is not None else np.zeros(len(data))
        for i, m in enumerate(data):
            name = CLASS_NAMES.get(int(cls[i]), f"class_{int(cls[i])}")
            classes[name] = classes.get(name, 0) + 1
            m_full = cv2.resize(m, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
            m_full = m_full[top:top + int(H * scale), left:left + int(W * scale)]
            m_back = cv2.resize(m_full, (W, H), interpolation=cv2.INTER_LINEAR)
            masks.append((m_back > 0.5).astype(np.uint8))
            mask_classes.append(name)

    if not masks:
        return {"masks": 0, "classes": {}, "footprint_xy": None, "note": "model nič nenašiel"}

    # pixely → svet (EPSG:3857) → EPSG:8353
    xmin, ymin, xmax, ymax = bbox3857
    t = Transformer.from_crs("EPSG:3857", "EPSG:8353", always_xy=True)
    union = np.zeros((H, W), dtype=np.uint8)
    for m in masks:
        union = np.maximum(union, m)

    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < 400:      # malé šumové plochy (v px)
            continue
        approx = cv2.approxPolyDP(c, 6.0, True).reshape(-1, 2)
        ring = []
        for px, py in approx:
            X = xmin + (px / W) * (xmax - xmin)
            Y = ymax - (py / H) * (ymax - ymin)
            x, y = t.transform(X, Y)
            ring.append([float(x), float(y)])
        if len(ring) >= 3:
            polys.append(Polygon(ring).buffer(0))

    if not polys:
        return {"masks": len(masks), "classes": classes, "footprint_xy": None,
                "note": "kontúry príliš malé"}

    u = unary_union(polys)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    ring = [[float(x), float(y)] for x, y in u.exterior.coords]

    # jednotlivé masky (pre segmentáciu riadenú modelom)
    detail = []
    for m, cname in zip(masks, mask_classes):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 400:
                continue
            approx = cv2.approxPolyDP(c, 5.0, True).reshape(-1, 2)
            poly = []
            for px, py in approx:
                X = xmin + (px / W) * (xmax - xmin)
                Y = ymax - (py / H) * (ymax - ymin)
                x, y = t.transform(X, Y)
                poly.append([float(x), float(y)])
            if len(poly) >= 3:
                detail.append({"class": cname, "polygon_xy": poly,
                               "area_px": float(cv2.contourArea(c))})

    return {
        "masks": len(masks),
        "classes": classes,
        "footprint_xy": ring,
        "area_m2": float(u.area),
        "masks_detail": detail,
        "source": "vision_model",
        "note": "obrys z vycvičeného segmentačného modelu (Vision) — používa sa, keď GIS nemá geometriu",
    }
