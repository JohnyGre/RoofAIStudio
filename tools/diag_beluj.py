"""
diag_beluj.py - Diagnostika segmentacie Beluj 50
Spusti: .venv\Scripts\python.exe tools/diag_beluj.py
"""
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- GPS Beluj 50 -> S-JTSK EPSG:5514 ---
LAT = 48.3518592
LON = 18.8923904

try:
    from pyproj import Transformer
    tf = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
    cx, cy = tf.transform(LON, LAT)
    print(f"Beluj 50 S-JTSK: x={cx:.2f}, y={cy:.2f}")
except ImportError:
    # Manualne hodnoty (predpocitane)
    cx, cy = -439720.0, -1269520.0
    print(f"pyproj chyba, pouzivam manualny odhad: x={cx:.2f}, y={cy:.2f}")

# --- Nacitanie LAZ ---
from app.core.pointcloud import load_laz_files, voxel_downsample, statistical_outlier_removal, classify_roof_points

LAZ_DIR = "data/laz"
laz_files = [
    os.path.join(LAZ_DIR, f)
    for f in os.listdir(LAZ_DIR)
    if "BanskaStiavnica" in f and f.endswith(".laz")
]
print(f"\nNacitavam {len(laz_files)} LAZ suborov pre Beluj...")

RADIUS = 80.0
pts_raw = load_laz_files(laz_files, classes=[6], center_xy=(cx, cy), radius=RADIUS)
print(f"Po class6 + radius {RADIUS}m: {len(pts_raw)} bodov")

pts = voxel_downsample(pts_raw, voxel_size=0.3)
print(f"Po voxel 0.3m: {len(pts)} bodov")

pts = statistical_outlier_removal(pts)
print(f"Po SOR: {len(pts)} bodov")

pts = classify_roof_points(pts, min_height=2.0)
print(f"Stresne body: {len(pts)} bodov")
print(f"Z rozsah: {pts[:,2].min():.2f} - {pts[:,2].max():.2f} m")

# --- 2D connected components - separacia budov ---
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

xy = pts[:, :2]
tree = cKDTree(xy)
pairs = tree.query_pairs(3.0, output_type="ndarray")
if len(pairs) > 0:
    row, col = pairs[:, 0], pairs[:, 1]
    graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(pts), len(pts)))
    n_comp, labels = connected_components(graph, directed=False)
    sizes = [(int(np.sum(labels == c)), c) for c in range(n_comp)]
    sizes.sort(reverse=True)
    print(f"\n2D CC budovy: {n_comp} komponentov")
    for i, (sz, c) in enumerate(sizes[:8]):
        mask = labels == c
        bpts = pts[mask]
        w = bpts[:,0].max() - bpts[:,0].min()
        h = bpts[:,1].max() - bpts[:,1].min()
        print(f"  Building_{i}: {sz} bodov, rozmery {w:.1f}x{h:.1f}m, "
              f"centrum ({bpts[:,0].mean():.1f}, {bpts[:,1].mean():.1f})")

    # Najblizsia budova k centru
    dists_to_center = []
    for sz, c in sizes:
        mask = labels == c
        bpts = pts[mask]
        cent = bpts[:, :2].mean(axis=0)
        d = np.linalg.norm(cent - np.array([cx, cy]))
        dists_to_center.append((d, sz, c))
    dists_to_center.sort()
    best_c = dists_to_center[0][2]
    roof_pts = pts[labels == best_c]
    print(f"\nVybrana budova: {len(roof_pts)} bodov (najblizsia k centru, dist={dists_to_center[0][0]:.1f}m)")
else:
    roof_pts = pts
    print("Ziadne pary pre CC - pouzivam vsetky body")

# --- RANSAC segmentacia ---
print("\n--- RANSAC segmentacia ---")
from app.plugins.exact_roof_planes import _multi_plane_ransac, _clean_boundary_polygon, _polygon_area_3d

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

raw_planes = _multi_plane_ransac(
    roof_pts,
    distance_threshold=0.06,
    min_inliers=200,
    max_planes=20,
    flat_roof_max_degree=8.0,
)
print(f"\nRANSAC nasiel {len(raw_planes)} rovin (pred low_conf filtrom)")

for i, p in enumerate(raw_planes):
    pts3d = roof_pts[p["point_idx"]]
    normal = p["normal"]
    slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
    centroid = pts3d.mean(axis=0)
    density = p.get("_density", 0)
    low_conf = p.get("_low_confidence", False)
    is_flat = p.get("_is_flat", False)
    
    # Polygon test
    try:
        poly = _clean_boundary_polygon(pts3d, normal)
        area = _polygon_area_3d(poly, normal)
        poly_arr = np.array(poly)
        poly_cent = poly_arr.mean(axis=0)
        poly_dist_from_pts = np.linalg.norm(poly_cent[:2] - centroid[:2])
        flag = " *** POLYGON DALEKO OD BODOV!" if poly_dist_from_pts > 5.0 else ""
    except Exception as e:
        area = 0.0
        poly_dist_from_pts = -1
        flag = f" *** POLYGON ERROR: {e}"

    status = "[LOW_CONF]" if low_conf else "[flat]" if is_flat else "      "
    print(f"  P{i:2d} {status}: {len(p['point_idx']):5d} bodov, sklon={slope:.1f}°, "
          f"area={area:.1f}m², density={density:.3f}, "
          f"cent=({centroid[0]:.1f},{centroid[1]:.1f},{centroid[2]:.1f}), "
          f"poly_dist={poly_dist_from_pts:.2f}m{flag}")

# Uloz body pre vizualizaciu
np.save("output/beluj_diag_roof_pts.npy", roof_pts)
print(f"\nUlozene stresne body -> output/beluj_diag_roof_pts.npy")
