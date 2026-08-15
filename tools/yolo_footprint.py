# -*- coding: utf-8 -*-
"""YOLO footprint -> S-JTSK polygon, porovnanie s LAZ budovou."""
import sys, os, json, math
sys.path.insert(0, r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio")

import numpy as np
import cv2
import torch
import pyproj

PROJ = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio"
OUT = os.path.join(PROJ, "output")

# --- Ortofoto geo kontext ---
lat, lon = 48.3961053, 17.587124
def to_merc(lat, lon):
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y
cx_m, cy_m = to_merc(lat, lon)
half = 100.0
xmin, xmax = cx_m - half, cx_m + half
ymin, ymax = cy_m - half, cy_m + half
SIZE = 4096
scale = SIZE / (xmax - xmin)

t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)

def px_to_sjtsk(px, py):
    mx = xmin + px / scale
    my = ymax - py / scale
    # Mercator -> WGS84
    lon2 = mx * 180.0 / 20037508.34
    lat2 = (math.atan(math.exp(my * math.pi / 180.0 / 20037508.34 * 180.0)) * 2 - math.pi / 2) * 180.0 / math.pi
    # WGS84 -> S-JTSK
    x5514, y5514 = t.transform(lon2, lat2)
    return x5514, y5514

# --- YOLO ---
from ultralytics import YOLO
model = YOLO(os.path.join(PROJ, "ai_models", "yolov8n_building_seg.pt"))
img_path = os.path.join(PROJ, "data", "ortho", "Atriova_16H_zbgis_200m.jpg")
results = model.predict(img_path, conf=0.25, device="cpu", verbose=False)
r = results[0]

# Ocakavana poloha nasej budovy (LAZ building_0 stred) v px
laz = np.load(os.path.join(PROJ, "data", "cache", "building_0.npy"))
laz_cx, laz_cy = laz[:, 0].mean(), laz[:, 1].mean()
# S-JTSK -> px (inverzna transformacia)
t_inv = pyproj.Transformer.from_crs("EPSG:5514", "EPSG:4326", always_xy=True)
lon_laz, lat_laz = t_inv.transform(laz_cx, laz_cy)
mx_laz, my_laz = to_merc(lat_laz, lon_laz)
exp_px = (mx_laz - xmin) * scale
exp_py = (ymax - my_laz) * scale
print(f"Ocakavana poloha (LAZ stred): px ({exp_px:.0f}, {exp_py:.0f})")

# Vyber detekciu najbližšie k ocakavanej polohe
best = None
best_dist = 1e18
for j, box in enumerate(r.boxes):
    if r.masks is None:
        continue
    x0, y0, x1, y1 = box.xyxy[0].tolist()
    bx, by = (x0 + x1) / 2, (y0 + y1) / 2
    dist = (bx - exp_px) ** 2 + (by - exp_py) ** 2
    if dist < best_dist:
        best_dist = dist
        best = j

if best is None:
    print("Ziadna maska!")
    sys.exit(1)

j = best
mask = r.masks[j].data[0].cpu().numpy().astype(np.uint8)
conf = float(r.boxes[j].conf[0])
print(f"Vybrata maska: [{j}] conf={conf:.3f}, dist={best_dist:.0f} px²")

# Maska je 640x640 (resized) - treba skalovat spat na 4096x4096
# Ultralytics masky su v pôvodnom rozlíšení? Kontrola: mask.shape = (640,640)
# Pravdepodobne su v rozliseni obrazka po resize. Zistime scale.
# r.orig_shape = (H, W) = (4096, 4096); mask je 640x640 -> scale = 4096/640
orig_h, orig_w = r.orig_shape
mh, mw = mask.shape
sx = orig_w / mw
sy = orig_h / mh
print(f"Maska {mw}x{mh}, orig {orig_w}x{orig_h}, scale ({sx:.1f}, {sy:.1f})")

# Upscale masky do orig rozlisenia
mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# Kontury -> polygon (najvacsi kontur)
contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest = max(contours, key=cv2.contourArea)
print(f"Kontur: {len(biggest)} bodov, area {cv2.contourArea(biggest):.0f} px")

# Zjednodus (approxPolyDP)
perim = cv2.arcLength(biggest, True)
poly = cv2.approxPolyDP(biggest, 0.01 * perim, True)
print(f"Po zjednoduseni: {len(poly)} bodov")

# Transformuj do S-JTSK
footprint = []
for pt in poly:
    px, py = float(pt[0][0]), float(pt[0][1])
    x5514, y5514 = px_to_sjtsk(px, py)
    footprint.append([round(x5514, 2), round(y5514, 2)])

# Vypis
xs = [p[0] for p in footprint]
ys = [p[1] for p in footprint]
print(f"\nYOLO footprint (S-JTSK):")
print(f"  X: {min(xs):.2f} .. {max(xs):.2f} ({(max(xs)-min(xs)):.1f} m)")
print(f"  Y: {min(ys):.2f} .. {max(ys):.2f} ({(max(ys)-min(ys)):.1f} m)")

# Porovnanie s LAZ building_0
print(f"\nLAZ building_0:")
print(f"  X: {laz[:,0].min():.2f} .. {laz[:,0].max():.2f} ({(laz[:,0].max()-laz[:,0].min()):.1f} m)")
print(f"  Y: {laz[:,1].min():.2f} .. {laz[:,1].max():.2f} ({(laz[:,1].max()-laz[:,1].min()):.1f} m)")

# Uloz footprint JSON
out_json = os.path.join(PROJ, "data", "exports", "atriova_16H_yolo_footprint.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "address": "Átriová 9309/16H, Trnava",
        "source": "yolov8n_building_seg.pt",
        "conf": conf,
        "crs": "EPSG:5514",
        "footprint": footprint,
    }, f, ensure_ascii=False, indent=2)
print(f"\nJSON: {out_json}")

# Uloz masku PNG (orezana na budovu)
x0, y0, x1, y1 = r.boxes[j].xyxy[0].tolist()
crop = mask_full[int(y0):int(y1), int(x0):int(x1)]
cv2.imencode(".png", crop * 255)[1].tofile(os.path.join(OUT, "atriova_16H_footprint_mask.png"))
print(f"Mask PNG: output/atriova_16H_footprint_mask.png")
