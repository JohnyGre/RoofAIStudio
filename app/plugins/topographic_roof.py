#!/usr/bin/env python3
"""
Topographic Roof Edge Detector — výškové rezy z LAZ bodového mračna.
Algoritmus:
  1. Z_min+δ → odkvap (α-shape vonkajšieho obrysu)
  2. Z_max-δ → hrebeň (fit línie najvyšších bodov)
  3. Horizontálne rezy každých ΔZ → sledovanie rohov → nárožie/úžľabie
  4. Zvislé hrany → štít
Výstup: 3D polygóny rovín + klasifikované hrany + kótovaný viewer.
"""
import numpy as np, math, json, os, glob
from collections import defaultdict, Counter
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay

# ─── Helpers ────────────────────────────────────────────────────────────────

def alpha_shape_2d(pts, alpha=None):
    """Concave hull with auto-alpha."""
    if len(pts) < 4:
        try: return pts[ConvexHull(pts).vertices]
        except: return pts
    try:
        import alphashape
        if alpha is None:
            tree = KDTree(pts); d,_ = tree.query(pts,k=5)
            alpha = float(np.percentile(d[:,1:].mean(axis=1), 90)) * 2
        shp = alphashape.alphashape(pts, alpha)
        if shp.geom_type == 'Polygon':
            return np.array(shp.exterior.coords)[:-1]
        elif shp.geom_type == 'MultiPolygon':
            return np.array(max(shp.geoms, key=lambda g:g.area).exterior.coords)[:-1]
    except: pass
    try: return pts[ConvexHull(pts).vertices]
    except: return pts

def simplify_polygon(pts, max_v=10):
    """Reduce vertex count, keeping shape."""
    n = len(pts)
    if n <= max_v: return pts
    step = max(1, n // max_v)
    return pts[::step][:max_v]

def fit_line_2d(points_2d):
    """Fit 2D line to points, return (start_pt, end_pt) along principal direction."""
    mean = points_2d.mean(axis=0)
    pca = PCA(n_components=2).fit(points_2d)
    direction = pca.components_[0]
    proj = points_2d.dot(direction)
    t_min, t_max = proj.min(), proj.max()
    return mean + t_min * direction, mean + t_max * direction

def find_contour_corners(contour_2d, angle_threshold=30):
    """Find corners (significant direction changes) in a 2D contour."""
    if len(contour_2d) < 4:
        return list(range(len(contour_2d)))
    n = len(contour_2d)
    corners = []
    for i in range(n):
        prev_i = (i-1)%n; next_i = (i+1)%n
        v1 = contour_2d[prev_i] - contour_2d[i]
        v2 = contour_2d[next_i] - contour_2d[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: continue
        cos_ang = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
        ang = math.degrees(math.acos(cos_ang))
        if ang < 180 - angle_threshold:
            corners.append(i)
    return sorted(set(corners))

def track_corner(pt_2d, next_slice_contour, search_radius=3.0):
    """Find matching corner in next Z-slice contour."""
    best_idx, best_dist = None, search_radius
    for j, cp in enumerate(next_slice_contour):
        d = np.linalg.norm(pt_2d - cp)
        if d < best_dist:
            best_dist = d; best_idx = j
    return best_idx, best_dist


# ─── Main Algorithm ─────────────────────────────────────────────────────────

def topographic_roof_analysis(lidar_pts_xyz, ground_z, dz_slice=0.5, delta_z=0.50):
    """
    Analyze roof from LiDAR point cloud using height slices.
    
    Args:
        lidar_pts_xyz: (N,3) array of building-class points
        ground_z: ground elevation
        dz_slice: vertical slice step (m)
        delta_z: thickness of eave/ridge bands (m)
    
    Returns:
        dict with planes and edge summary
    """
    pts = lidar_pts_xyz.copy()
    pts[:, 2] -= ground_z  # relative height
    
    # Filter to roof-level points (> 0.5m above ground)
    roof = pts[pts[:, 2] > 0.5]
    if len(roof) < 50: return {'planes': [], 'edges_summary': {}}
    
    z_min, z_max = roof[:, 2].min(), roof[:, 2].max()
    centroid_xy = roof[:, :2].mean(axis=0)
    
    # ═══ 1. EAVE (odkvap) ═══
    # Use ConvexHull of ALL roof points for full building footprint
    try:
        hull = ConvexHull(roof[:, :2])
        eave_poly_2d = roof[:, :2][hull.vertices]
    except:
        eave_poly_2d = roof[:, :2]
    eave_poly_2d = simplify_polygon(eave_poly_2d, max_v=20)
    eave_z = roof[roof[:, 2] <= z_min + delta_z][:,2].mean() if len(roof[roof[:,2]<=z_min+delta_z]) > 0 else z_min
    
    # ═══ 2. RIDGE (hrebeň) ═══
    ridge_pts = roof[roof[:, 2] >= z_max - delta_z]
    if len(ridge_pts) < 20:
        ridge_pts = roof[roof[:, 2] >= z_max - 1.0]
    ridge_lines = []
    if len(ridge_pts) >= 10:
        # Cluster ridge points spatially
        from sklearn.cluster import DBSCAN
        ridge_xy = ridge_pts[:, :2]
        ridge_labels = DBSCAN(eps=2.0, min_samples=8).fit_predict(ridge_xy)
        for cid in sorted(set(ridge_labels)):
            if cid == -1: continue
            cpts = ridge_xy[ridge_labels == cid]
            if len(cpts) < 5: continue
            p1, p2 = fit_line_2d(cpts)
            ridge_lines.append({
                'start': [round(float(p1[0]),3), round(float(p1[1]),3), round(float(ridge_pts[ridge_labels==cid,2].mean()),3)],
                'end':   [round(float(p2[0]),3), round(float(p2[1]),3), round(float(ridge_pts[ridge_labels==cid,2].mean()),3)],
            })
    
    # ═══ 3. SLICE ANALYSIS ═══
    z_levels = np.arange(z_min + delta_z, z_max - delta_z, dz_slice)
    z_levels = np.append(z_levels, [z_max - delta_z])  # add top slice
    
    slice_data = []  # list of {z, contour_2d, corners_2d, corner_indices}
    
    for z_level in z_levels:
        slice_pts = roof[(roof[:, 2] >= z_level) & (roof[:, 2] <= z_level + delta_z * 2)]
        if len(slice_pts) < 15: continue
        
        try:
            hull = ConvexHull(slice_pts[:, :2])
            contour_2d = slice_pts[:, :2][hull.vertices]
        except:
            contour_2d = slice_pts[:, :2]
        if len(contour_2d) < 4: continue
        contour_2d = simplify_polygon(contour_2d, max_v=16)
        
        corners_idx = find_contour_corners(contour_2d, angle_threshold=15)
        corners_2d = [contour_2d[i] for i in corners_idx]
        
        slice_data.append({
            'z': round(float(z_level + delta_z), 3),
            'contour': contour_2d,
            'n_pts': len(slice_pts),
            'corners_idx': corners_idx,
            'corners_2d': corners_2d,
        })
    
    if len(slice_data) < 2:
        slice_data = [{'z': round(float(z_min),3), 'contour': eave_poly_2d, 'corners_2d': eave_poly_2d, 'n_pts': len(eave_pts)}]
    
    # ═══ 4. CORNER TRACKING → narozie/uzlabie ═══
    valley_ridge_edges = []  # (n)arozie or (u)zlabie
    gable_edges = []         # (f)stit
    
    if len(slice_data) >= 2:
        # Use lowest slice as reference
        bottom = slice_data[0]
        bottom_corners = np.array(bottom['corners_2d']) if len(bottom['corners_2d']) > 2 else bottom['contour']
        
        # Track each corner upward
        for ci, bc in enumerate(bottom_corners):
            path = [(bc[0], bc[1], bottom['z'])]  # 3D track
            current_pt = bc.copy()
            
            for sl in slice_data[1:]:
                sl_corners = np.array(sl['corners_2d']) if len(sl['corners_2d']) > 0 else np.array([])
                if len(sl_corners) > 0:
                    match_idx, match_dist = track_corner(current_pt, sl_corners, search_radius=5.0)
                    if match_idx is not None:
                        current_pt = sl_corners[match_idx]
                        path.append((current_pt[0], current_pt[1], sl['z']))
            
            # Create single edge from full path (bottom to top)
            seg_start = np.array(path[0])
            seg_end   = np.array(path[-1])
            full_len = np.linalg.norm(seg_end - seg_start)
            if full_len < 0.5: continue
            d_s = np.linalg.norm(seg_start[:2] - centroid_xy)
            d_e = np.linalg.norm(seg_end[:2] - centroid_xy)
            if d_e < d_s - 0.3: seg_type = 'n'
            elif d_e > d_s + 0.3: seg_type = 'u'
            else: seg_type = 'f'
            valley_ridge_edges.append({
                'type': seg_type,
                'vertices_3d': [[round(float(seg_start[0]),3), round(float(seg_start[1]),3), round(float(seg_start[2]),3)],
                                 [round(float(seg_end[0]),3), round(float(seg_end[1]),3), round(float(seg_end[2]),3)]],
                'len_3d': round(float(full_len), 3),
            })

    # ═══ 5. BUILD PLANES ═══
    planes = []
    etypes = {'o':0,'n':0,'u':0,'h':0,'f':0}
    esums = {}
    
    # --- Eave polygon as plane P0 (platform-like, but offset from ground) ---
    eave_verts_3d = [[round(float(x),3), round(float(y),3), round(float(eave_z),3)] for x,y in eave_poly_2d]
    eave_edges = []
    for ei in range(len(eave_verts_3d)):
        v1, v2 = np.array(eave_verts_3d[ei]), np.array(eave_verts_3d[(ei+1)%len(eave_verts_3d)])
        elen = float(np.linalg.norm(v2-v1))
        if elen < 0.3: continue
        nid = etypes['o']; etypes['o'] += 1
        eave_edges.append({'id':'o{}'.format(nid+1),'type':'o','length_m':round(elen,3),'v1':ei,'v2':(ei+1)%len(eave_verts_3d),'start':eave_verts_3d[ei],'end':eave_verts_3d[(ei+1)%len(eave_verts_3d)]})
        esums.setdefault('o',[]).append(round(elen,3))
    
    try:
        from shapely.geometry import Polygon
        eave_area = Polygon(eave_poly_2d).area
    except:
        eave_area = float(ConvexHull(eave_poly_2d).volume)
    
    planes.append({
        'id':'O1','type':'odkvap_obrys','area_m2':round(float(eave_area),2),
        'pitch_deg':0.0,'z_min_m':round(float(z_min),2),'z_max_m':round(float(z_max),2),
        'vertices_3d':eave_verts_3d,'edges':eave_edges,
    })
    
    # --- Ridge segments ---
    for ri, rl in enumerate(ridge_lines):
        rlen = round(float(np.linalg.norm(np.array(rl['end']) - np.array(rl['start']))), 3)
        if rlen < 0.3: continue
        nid = etypes['h']; etypes['h'] += 1
        planes.append({
            'id':'H{}'.format(ri+1),'type':'hreben',
            'area_m2':0,'pitch_deg':0.0,
            'z_min_m':round(float(rl['start'][2]),2),'z_max_m':round(float(rl['end'][2]),2),
            'vertices_3d':[rl['start'], rl['end']],
            'edges':[{'id':'h{}'.format(nid+1),'type':'h','length_m':rlen,'v1':0,'v2':1,'start':rl['start'],'end':rl['end']}],
        })
        esums.setdefault('h',[]).append(rlen)
    
    # --- Valley/Ridge/Stit edges ---
    for vri, vr in enumerate(valley_ridge_edges):
        ett = vr['type']
        nid = etypes[ett]; etypes[ett] += 1
        planes.append({
            'id':'{}{}'.format(ett.upper(), vri+1),'type':{'n':'narozie','u':'uzlabie','f':'stit'}[ett],
            'area_m2':0,'pitch_deg':0.0,
            'z_min_m':round(float(vr['vertices_3d'][0][2]),2),
            'z_max_m':round(float(vr['vertices_3d'][-1][2]),2),
            'vertices_3d':vr['vertices_3d'],
            'edges':[{'id':'{}{}'.format(ett,nid+1),'type':ett,'length_m':vr['len_3d'],'v1':0,'v2':len(vr['vertices_3d'])-1,'start':vr['vertices_3d'][0],'end':vr['vertices_3d'][-1]}],
        })
        esums.setdefault(ett,[]).append(vr['len_3d'])
    
    # ═══ 6. SUMMARY ═══
    tn = {'o':'odkvap','n':'nárožie','u':'úžľabie','h':'hrebeň','f':'štít'}
    summary = {et:{'nazov':tn.get(et,et),'pocet':len(lengths),'celkom_m':round(sum(lengths),3)} for et,lengths in esums.items()}
    
    return {
        'ground_z': ground_z,
        'z_min': round(float(z_min),2),
        'z_max': round(float(z_max),2),
        'centroid_xy': [round(float(centroid_xy[0]),3), round(float(centroid_xy[1]),3)],
        'n_slices': len(slice_data),
        'ridge_count': len(ridge_lines),
        'planes': planes,
        'edges_summary': summary,
    }
