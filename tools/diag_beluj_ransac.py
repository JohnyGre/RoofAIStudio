"""
diag_beluj_ransac.py - RANSAC test priamo na beluj_50_mesh.obj
"""
import numpy as np, sys, logging, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

import trimesh
mesh = trimesh.load("output/beluj_50_mesh.obj", process=False)
pts = np.asarray(mesh.vertices)
print(f"Mesh bodov: {len(pts)}")
print(f"Z rozsah: {pts[:,2].min():.2f} - {pts[:,2].max():.2f}")

from app.plugins.exact_roof_planes import (
    _multi_plane_ransac, _clean_boundary_polygon, _polygon_area_3d, mesh_to_roof_planes_exact
)

# Test 1: znizena min_inliers (mesh ma len 296 vrcholov)
print("\n--- Test 1: min_inliers=20 ---")
raw_planes = _multi_plane_ransac(
    pts,
    distance_threshold=0.06,
    min_inliers=20,
    max_planes=20,
    flat_roof_max_degree=8.0,
)
print(f"RANSAC nasiel {len(raw_planes)} rovin")

for i, p in enumerate(raw_planes):
    pts3d = pts[p["point_idx"]]
    normal = p["normal"]
    slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
    centroid = pts3d.mean(axis=0)
    low_conf = p.get("_low_confidence", False)
    is_flat = p.get("_is_flat", False)

    try:
        poly = _clean_boundary_polygon(pts3d, normal)
        area = _polygon_area_3d(poly, normal)
        poly_arr = np.array(poly)
        poly_cent = poly_arr.mean(axis=0)
        poly_dist = np.linalg.norm(poly_cent[:2] - centroid[:2])
        flag = " *** POLYGON DALEKO!" if poly_dist > 5.0 else ""
    except Exception as e:
        area, poly_dist = 0.0, -1
        flag = f" *** ERR: {e}"

    status = "[LOW]" if low_conf else "[flat]" if is_flat else "     "
    print(f"  P{i:2d} {status}: {len(p['point_idx']):5d} bodov, sklon={slope:.1f}deg, "
          f"area={area:.1f}m2, poly_dist={poly_dist:.2f}m{flag}")

# Test 2: full pipeline cez mesh_to_roof_planes_exact
print("\n--- Test 2: mesh_to_roof_planes_exact (full pipeline) ---")
logging.getLogger().setLevel(logging.INFO)
try:
    planes = mesh_to_roof_planes_exact(
        "output/beluj_50_mesh.obj",
        distance_threshold=0.06,
        min_inliers=20,
        max_planes=20,
    )
    print(f"Vysledok: {len(planes)} rovin")
    for pl in planes:
        n_verts = len(pl.get("vertices", []))
        n_edges = len(pl.get("edges", []))
        low = pl.get("low_confidence", False)
        print(f"  {pl['id']}: {pl['type']}, {pl['pitch_deg']}deg, "
              f"{pl['area_m2']}m2, {n_verts} vrcholov, {n_edges} hran, low_conf={low}")
except Exception as e:
    print(f"CHYBA: {e}")
    import traceback
    traceback.print_exc()
