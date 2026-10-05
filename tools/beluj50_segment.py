"""
beluj50_segment.py - Kompletna segmentacia strechy Beluj 50
Presne GPS: 48.351930, 18.892250
"""
import numpy as np
import os, sys, json, logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 1. GPS → S-JTSK
# ─────────────────────────────────────────
# -----------------------------------------
# 1. GPS -> S-JTSK
# -----------------------------------------
LAT = 48.351930
LON = 18.892250

from pyproj import Transformer
tf = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
cx, cy = tf.transform(LON, LAT)
print(f"\n[1] GPS {LAT}, {LON} -> S-JTSK ({cx:.2f}, {cy:.2f})")

# -----------------------------------------
# 2. Nacitanie LAZ
# -----------------------------------------
from app.core.pointcloud import load_laz_files, voxel_downsample, statistical_outlier_removal, classify_roof_points
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

LAZ_DIR = "data/laz"
laz_files = sorted([
    os.path.join(LAZ_DIR, f)
    for f in os.listdir(LAZ_DIR)
    if "BanskaStiavnica" in f and f.endswith(".laz")
])
print(f"\n[2] Nacitavam {len(laz_files)} LAZ suborov (radius=60m od GPS)...")

pts_raw = load_laz_files(laz_files, classes=[6], center_xy=(cx, cy), radius=60.0)
print(f"    class6 + radius 60m: {len(pts_raw)} bodov")

pts = voxel_downsample(pts_raw, voxel_size=0.25)
print(f"    voxel 0.25m: {len(pts)} bodov")

pts = statistical_outlier_removal(pts, k=15, std_ratio=2.0)
print(f"    SOR: {len(pts)} bodov")

pts = classify_roof_points(pts, min_height=2.0)
print(f"    stresne body (>2m): {len(pts)} bodov, Z={pts[:,2].min():.1f}-{pts[:,2].max():.1f}m")

# -----------------------------------------
# 3. 2D CC -> vyber budovy Beluj 50
# -----------------------------------------
print("\n[3] Separacia budov (2D CC, radius=2.5m)...")
tree = cKDTree(pts[:, :2])
pairs = tree.query_pairs(2.5, output_type="ndarray")
if len(pairs) == 0:
    print("    CHYBA: ziadne pary CC")
    sys.exit(1)

row, col = pairs[:, 0], pairs[:, 1]
graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(pts), len(pts)))
n_comp, labels = connected_components(graph, directed=False)

center = np.array([cx, cy])
print(f"    CC komponentov: {n_comp}")

all_bldgs = []
for c in range(n_comp):
    mask = labels == c
    sz = int(mask.sum())
    if sz < 30:
        continue
    bpts = pts[mask]
    cent = bpts[:, :2].mean(axis=0)
    d = np.linalg.norm(cent - center)
    all_bldgs.append((d, sz, c, cent, bpts))

all_bldgs.sort()
print(f"    Budovy (>=30 bodov):")
for i, (d, sz, c, cent, bpts) in enumerate(all_bldgs[:10]):
    w = bpts[:,0].max() - bpts[:,0].min()
    h = bpts[:,1].max() - bpts[:,1].min()
    print(f"      B{i}: dist={d:.1f}m, {sz} bodov, {w:.1f}x{h:.1f}m, cent=({cent[0]:.1f},{cent[1]:.1f})")

# Vyber najblizsiu budovu s dostatkom bodov (min 200)
selected = None
for d, sz, c, cent, bpts in all_bldgs:
    if sz >= 200:
        selected = (d, sz, c, cent, bpts)
        break

if selected is None:
    print("    WARN: Ziadna budova >=200 bodov, beru najblizsiu >=50...")
    for d, sz, c, cent, bpts in all_bldgs:
        if sz >= 50:
            selected = (d, sz, c, cent, bpts)
            break

if selected is None:
    print("CHYBA: Ziadna vhodna budova!")
    sys.exit(1)

d, sz, c, cent, roof_pts = selected
print(f"\n    OK Vybrana budova: {sz} bodov, dist={d:.1f}m, cent=({cent[0]:.1f},{cent[1]:.1f})")

# -----------------------------------------
# 4. RANSAC segmentacia
# -----------------------------------------
print("\n[4] RANSAC multi-plane segmentacia...")

from app.plugins.exact_roof_planes import (
    _multi_plane_ransac, _clean_boundary_polygon, _polygon_area_3d,
    _confirmed_adjacent_planes, _dihedral_type, clip_polygon_by_plane,
    mesh_to_roof_planes_exact
)
from collections import defaultdict

# Parametre
DIST_THRESH = 0.08       # tolerancia RANSAC (m)
MIN_INLIERS_PT = max(50, sz // 30)  # adaptivne podla poctu bodov
print(f"    min_inliers={MIN_INLIERS_PT}, dist_thresh={DIST_THRESH}")

raw_planes = _multi_plane_ransac(
    roof_pts,
    distance_threshold=DIST_THRESH,
    min_inliers=MIN_INLIERS_PT,
    max_planes=20,
    flat_roof_max_degree=8.0,
)
print(f"    RANSAC: {len(raw_planes)} rovin (pred filtrovanim)")

# Detailny vypis rovin
for i, p in enumerate(raw_planes):
    pts3d = roof_pts[p["point_idx"]]
    normal = p["normal"]
    slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
    centroid = pts3d.mean(axis=0)
    density = p.get("_density", 0)
    low_conf = p.get("_low_confidence", False)
    is_flat = p.get("_is_flat", False)
    status = "[LOW_CONF]" if low_conf else "[plocha] " if is_flat else "         "
    print(f"    P{i:2d} {status}: {len(p['point_idx']):5d} bodov, sklon={slope:.1f}deg, "
          f"density={density:.4f}, cent_z={centroid[2]:.1f}m")

# -----------------------------------------
# 5. Polygony + susednost + orezanie
# -----------------------------------------
print("\n[5] Polygony, susednost, orezanie...")

for p in raw_planes:
    pts3d = roof_pts[p["point_idx"]]
    p["_poly"] = _clean_boundary_polygon(pts3d, p["normal"], 0.02)
    p["_centroid"] = np.mean(p["_poly"], axis=0)
    p["d"] = -p["normal"].dot(p["_centroid"])

good_idx = [i for i, p in enumerate(raw_planes) if not p.get("_low_confidence")]
low_conf_idx = [i for i in range(len(raw_planes)) if i not in good_idx]
print(f"    Good roviny: {len(good_idx)}, Low-conf: {len(low_conf_idx)}")

adjacency = _confirmed_adjacent_planes(
    roof_pts, raw_planes,
    near_tol=0.60,      # mierne volnejsia tolerancia pre vacsie budovy
    min_near_points=5,  # znižené (menej bodov ako Atriova)
    min_length_m=1.5,
)
adjacency = {k: v for k, v in adjacency.items() if k[0] in good_idx and k[1] in good_idx}
print(f"    Overenych susedstiev: {len(adjacency)}")

if len(adjacency) == 0:
    print("    WARN: Ziadne susedstvia - skusam s uvolnenejsimi parametrami...")
    adjacency = _confirmed_adjacent_planes(
        roof_pts, raw_planes,
        near_tol=1.0,
        min_near_points=3,
        min_length_m=0.8,
    )
    adjacency = {k: v for k, v in adjacency.items() if k[0] in good_idx and k[1] in good_idx}
    print(f"    Overenych susedstiev (uvolnene): {len(adjacency)}")

neighbors = defaultdict(set)
for (i, j) in adjacency:
    neighbors[i].add(j)
    neighbors[j].add(i)

# Orezanie polygonov
MIN_RETAINED = 0.20
final_polys = {}
for idx in good_idx:
    poly = list(raw_planes[idx]["_poly"])
    for nb in neighbors[idx]:
        q_normal = raw_planes[nb]["normal"]
        q_centroid = raw_planes[nb]["_centroid"]
        area_before = _polygon_area_3d(poly, raw_planes[idx]["normal"])
        cp = clip_polygon_by_plane(poly, q_centroid, q_normal, keep_positive=True)
        cn = clip_polygon_by_plane(poly, q_centroid, q_normal, keep_positive=False)
        ap = _polygon_area_3d(cp, raw_planes[idx]["normal"]) if len(cp) >= 3 else 0.0
        an = _polygon_area_3d(cn, raw_planes[idx]["normal"]) if len(cn) >= 3 else 0.0
        clipped = cp if ap >= an else cn
        area_after = max(ap, an)
        if area_before > 1e-6 and (area_after / area_before) < MIN_RETAINED:
            print(f"    WARN: Orez P{idx}<->P{nb} odstranil >{(1-area_after/area_before)*100:.0f}% -> preskoceny")
            clipped = poly
        poly = clipped if len(clipped) >= 3 else poly
    final_polys[idx] = poly

# -----------------------------------------
# 6. Zostav vysledok
# -----------------------------------------
print("\n[6] Zostav vysledok...")

_TYPE_CODE = {"okap": "o", "stit": "s", "hreben": "h", "narozie": "n", "uzlabie": "u"}
FLAT_DEG = 8.0

planes_out = []
for out_i, idx in enumerate(good_idx):
    poly = final_polys[idx]
    normal = raw_planes[idx]["normal"]
    slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))

    edges = []
    n = len(poly)
    for ei in range(n):
        v1, v2 = poly[ei], poly[(ei + 1) % n]
        length_m = float(np.linalg.norm(np.array(v2) - np.array(v1)))
        if length_m < 0.3:
            continue
        mid = tuple((np.array(v1) + np.array(v2)) / 2)
        edge_type = "okap"
        is_exact = False
        for nb in neighbors[idx]:
            key = (min(idx, nb), max(idx, nb))
            pi = adjacency.get(key)
            if pi is None:
                continue
            rel = np.array(mid) - pi["point"]
            t = rel @ pi["direction"]
            perp = rel - t * pi["direction"]
            if np.linalg.norm(perp) < 0.40 or (np.linalg.norm(perp) < 0.80 and pi["t_lo"] - 0.5 <= t <= pi["t_hi"] + 0.5):
                kind, z_var = _dihedral_type(pi, raw_planes[idx], raw_planes[nb])
                edge_type = kind
                is_exact = True
                break
        code = _TYPE_CODE[edge_type]
        edges.append({
            "id": f"{code}{ei+1}",
            "type": code,
            "length_m": round(length_m, 3),
            "start": [round(v, 3) for v in v1],
            "end": [round(v, 3) for v in v2],
            "exact": is_exact,
        })

    u = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u_vec = np.cross(normal, u); u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(normal, u_vec)
    c = np.mean(poly, axis=0)
    local2d = np.array([[(np.array(p) - c) @ u_vec, (np.array(p) - c) @ v_vec] for p in poly])
    x, y = local2d[:, 0], local2d[:, 1]
    area = abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0

    roof_type = "plocha" if slope < FLAT_DEG else "sedlova" if 15 <= slope <= 45 else "valbova"

    planes_out.append({
        "id": f"R{out_i + 1}",
        "type": roof_type,
        "area_m2": round(float(area), 2),
        "pitch_deg": round(slope, 1),
        "vertices": [[round(v, 3) for v in p] for p in poly],
        "edges": edges,
        "low_confidence": False,
    })

for idx in low_conf_idx:
    pts3d = roof_pts[raw_planes[idx]["point_idx"]]
    normal = raw_planes[idx]["normal"]
    slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
    planes_out.append({
        "id": f"R{len(planes_out)+1}_LOW_CONFIDENCE",
        "type": "neurcite",
        "area_m2": 0.0,
        "pitch_deg": round(slope, 1),
        "vertices": [],
        "edges": [],
        "low_confidence": True,
    })

# -----------------------------------------
# 7. Vypis + ulozenie
# -----------------------------------------
print(f"\n[7] Vysledok: {len(planes_out)} rovin")
total_area = 0.0
for pl in planes_out:
    n_e = len(pl.get("edges", []))
    n_v = len(pl.get("vertices", []))
    edge_types = {}
    for e in pl.get("edges", []):
        edge_types[e["type"]] = edge_types.get(e["type"], 0) + 1
    et_str = " ".join(f"{k}:{v}" for k, v in sorted(edge_types.items()))
    low = "WARN" if pl["low_confidence"] else "  "
    print(f"  {low} {pl['id']:20s}: {pl['type']:8s} {pl['pitch_deg']:5.1f}deg "
          f"{pl['area_m2']:7.1f}m2, {n_v}v {n_e}e [{et_str}]")
    if not pl["low_confidence"]:
        total_area += pl["area_m2"]

print(f"\n  Celkova plocha dobrych rovin: {total_area:.1f} m2")

# Ulozenie JSON
out_json = "output/beluj_50_planes.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({"address": "Beluj 50, 969 01 Beluj", "planes": planes_out}, f, ensure_ascii=False, indent=2)
print(f"\n  OK Ulozene: {out_json}")

# Ulozenie bodov pre debug
np.save("output/beluj_50_roof_pts.npy", roof_pts)
print(f"  OK Debug body: output/beluj_50_roof_pts.npy")
