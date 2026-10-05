"""
beluj50_viewer.py - Generovanie 3D vieweru pre Beluj 50
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Nacitaj planes
with open("output/beluj_50_planes.json", encoding="utf-8") as f:
    data = json.load(f)

planes = data["planes"]
print(f"Nacitanych {len(planes)} rovin")

# Generuj viewer cez existing template
from app.plugins.exact_roof_planes import write_exact_viewer_html

try:
    out = write_exact_viewer_html(
        planes=planes,
        address="Beluj 50, 969 01 Beluj",
        output_path="output/beluj_50_viewer.html",
        template_path="output/_triov_v9_viewer.html",
    )
    print(f"OK Viewer: {out}")
except Exception as e:
    print(f"CHYBA viewer: {e}")

# Vizualizacia top-down
import numpy as np, cv2

# Nacitaj roof_pts
roof_pts = np.load("output/beluj_50_roof_pts.npy")
print(f"Debug body: {len(roof_pts)}")

# Farebna mapa podla Z
z_min, z_max = roof_pts[:, 2].min(), roof_pts[:, 2].max()
xmin, xmax = roof_pts[:, 0].min() - 2, roof_pts[:, 0].max() + 2
ymin, ymax = roof_pts[:, 1].min() - 2, roof_pts[:, 1].max() + 2

SCALE = 8  # pixlov na meter
W = int((xmax - xmin) * SCALE) + 1
H = int((ymax - ymin) * SCALE) + 1
img = np.zeros((H, W, 3), dtype=np.uint8)

# Nakresli body
for pt in roof_pts:
    px = int((pt[0] - xmin) * SCALE)
    py = H - int((pt[1] - ymin) * SCALE) - 1
    t = (pt[2] - z_min) / max(z_max - z_min, 0.01)
    color = (int(255 * (1 - t)), int(100 + 155 * t), int(255 * t))  # blue->green->red
    if 0 <= px < W and 0 <= py < H:
        cv2.circle(img, (px, py), 1, color, -1)

# Nakresli polygony rovín
COLORS = [
    (255, 50, 50), (50, 255, 50), (50, 50, 255), (255, 255, 50),
    (255, 50, 255), (50, 255, 255), (200, 100, 50), (100, 200, 50),
    (200, 50, 100), (100, 50, 200), (50, 100, 200),
]

for pi, pl in enumerate(planes):
    if pl.get("low_confidence") or not pl.get("vertices"):
        continue
    verts = pl["vertices"]
    color = COLORS[pi % len(COLORS)]

    # Vyplnenie polygonu
    pts2d = []
    for v in verts:
        px = int((v[0] - xmin) * SCALE)
        py = H - int((v[1] - ymin) * SCALE) - 1
        pts2d.append([px, py])

    if len(pts2d) >= 3:
        pts_arr = np.array(pts2d, dtype=np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_arr], color)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        cv2.polylines(img, [pts_arr], True, color, 2)

    # Popis
    cx_px = int(np.mean([p[0] for p in pts2d]))
    cy_px = int(np.mean([p[1] for p in pts2d]))
    label = f"{pl['id']} {pl['pitch_deg']}deg"
    cv2.putText(img, label, (cx_px - 20, cy_px), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

# Info text
cv2.putText(img, "Beluj 50 - segmentacia rovin", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(img, f"{len([p for p in planes if not p.get('low_confidence')])} rovin", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

cv2.imwrite("output/beluj_50_topdown_v2.png", img)
print("OK Top-down: output/beluj_50_topdown_v2.png")
