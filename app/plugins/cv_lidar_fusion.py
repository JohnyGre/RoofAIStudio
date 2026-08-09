#!/usr/bin/env python3
"""YOLO + LiDAR fusion: YOLO gives clean 2D polygons, LiDAR gives precise Z heights."""
import numpy as np, json, os, glob
from pyproj import Transformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import math

def load_lidar_roof(laz_dir, lat, lon, radius=25):
    """Load building + ground points from LAZ files."""
    t = Transformer.from_crs('EPSG:4326', 'EPSG:8353', always_xy=True)
    te, tn = t.transform(lon, lat)
    bp, gp = [], []
    for f in glob.glob(os.path.join(laz_dir, '*.laz')):
        if '.copc.' in f: continue
        try:
            import laspy
            las = laspy.read(f)
            xs, ys, zs = np.array(las.x), np.array(las.y), np.array(las.z)
            cls = np.array(las.classification, dtype=np.uint8)
            d = np.sqrt((xs-te)**2 + (ys-tn)**2)
            n = d < radius
            if np.any(n & (cls==6)): bp.append(np.column_stack([xs[n&(cls==6)], ys[n&(cls==6)], zs[n&(cls==6)]]))
            if np.any(n & (cls==2)): gp.append(np.column_stack([xs[n&(cls==2)], ys[n&(cls==2)], zs[n&(cls==2)]]))
        except: pass
    if not bp: raise RuntimeError('No building points found')
    return np.vstack(bp), np.vstack(gp)

def get_plane_z_from_lidar(polygon_xy_m, lidar_pts, ground_z):
    """Sample Z heights from LiDAR for a polygon, return plane params."""
    from shapely.geometry import Point, Polygon
    poly = Polygon(polygon_xy_m)
    if not poly.is_valid: poly = poly.buffer(0)
    
    # Get LiDAR points inside polygon
    inside = []
    for pt in lidar_pts:
        if poly.contains(Point(pt[0], pt[1])):
            inside.append(pt)
    
    if len(inside) < 5:
        # Fallback: use nearby points
        tree = KDTree(lidar_pts[:, :2])
        centroid = np.mean(polygon_xy_m, axis=0)
        dists, idxs = tree.query([centroid], k=min(50, len(lidar_pts)))
        inside = lidar_pts[idxs[0]]
    
    inside = np.array(inside)
    
    # Fit plane
    pca = PCA(n_components=3).fit(inside)
    normal = pca.components_[2]
    if normal[2] < 0: normal = -normal
    pitch = math.degrees(math.acos(min(1, abs(normal[2]))))
    
    # Heights at each vertex
    vertex_zs = []
    for v in polygon_xy_m:
        # Find nearest LiDAR points
        dists, idxs = tree.query([v], k=10)
        nearby_z = lidar_pts[idxs[0], 2] - ground_z
        # Interpolate: use plane fit if available
        z = inside[:, 2].mean() - ground_z if len(inside) > 0 else nearby_z.mean()
        vertex_zs.append(round(float(z), 3))
    
    return {
        'pitch_deg': round(float(pitch), 1),
        'z_min': round(float(inside[:, 2].min() - ground_z), 2) if len(inside) > 0 else 0,
        'z_max': round(float(inside[:, 2].max() - ground_z), 2) if len(inside) > 0 else 0,
        'vertex_zs': vertex_zs,
        'point_count': len(inside),
    }

def classify_edge_type(v1, v2, zmin, zmax, pitch):
    """Edge: o=odkvap, h=hreben, f=stit."""
    dz = abs(v2[2] - v1[2])
    mz = (v1[2] + v2[2]) / 2
    if pitch < 5: return 'p'
    if mz <= zmin + 0.3: return 'o'  # Bottom = eave
    if mz >= zmax - 0.3: return 'h'  # Top = ridge
    return 'f'  # Rake/gable

def process(polygons_from_yolo, lidar_pts, ground_z, px_per_m):
    """Main fusion pipeline."""
    # Convert YOLO pixel polygons to meters
    planes = []
    all_edges = {}
    
    for i, yolo_plane in enumerate(polygons_from_yolo):
        contour = yolo_plane.get('contour', [])
        if len(contour) < 3: continue
        
        # Convert to meters
        poly_xy_m = []
        for pt in contour:
            if isinstance(pt[0], (list, tuple)):
                px, py = pt[0][0], pt[0][1]
            else:
                px, py = pt[0], pt[1]
            poly_xy_m.append((px / px_per_m, py / px_per_m))
        
        # Get Z from LiDAR
        lidar_info = get_plane_z_from_lidar(poly_xy_m, lidar_pts, ground_z)
        
        # Build 3D vertices
        verts_3d = []
        for vi, (x, y) in enumerate(poly_xy_m):
            z = lidar_info['vertex_zs'][vi] if vi < len(lidar_info['vertex_zs']) else lidar_info['z_min']
            verts_3d.append([round(x, 3), round(y, 3), round(z, 3)])
        
        zs = [v[2] for v in verts_3d]
        zmin, zmax = min(zs), max(zs)
        
        # Area in 3D
        from shapely.geometry import Polygon
        poly2d = Polygon(poly_xy_m)
        area_2d = poly2d.area
        pitch_rad = math.radians(lidar_info['pitch_deg'])
        area_3d = area_2d / max(math.cos(pitch_rad), 0.01)
        
        # Perimeter
        perim = 0
        for vi in range(len(verts_3d)):
            v1 = np.array(verts_3d[vi])
            v2 = np.array(verts_3d[(vi+1)%len(verts_3d)])
            perim += float(np.linalg.norm(v2 - v1))
        
        # Edge classification
        etypes = {'o': 0, 'n': 0, 'u': 0, 'h': 0, 'f': 0}
        edges = []
        for vi in range(len(verts_3d)):
            v1 = np.array(verts_3d[vi])
            v2 = np.array(verts_3d[(vi+1)%len(verts_3d)])
            elen = float(np.linalg.norm(v2 - v1))
            if elen < 0.2: continue
            
            etype = classify_edge_type(v1, v2, zmin, zmax, lidar_info['pitch_deg'])
            nid = etypes[etype]
            etypes[etype] += 1
            edges.append({
                'id': '{}{}'.format(etype, nid+1),
                'type': etype,
                'length_m': round(elen, 3),
                'v1': vi,
                'v2': (vi+1) % len(verts_3d),
            })
            all_edges.setdefault(etype, []).append(round(elen, 3))
        
        planes.append({
            'id': 'R{}'.format(i+1),
            'type': 'sedlova' if 15 <= lidar_info['pitch_deg'] <= 45 else 'valbova',
            'area_m2': round(float(area_3d), 2),
            'perimeter_m': round(perim, 2),
            'pitch_deg': lidar_info['pitch_deg'],
            'z_min_m': round(zmin, 2),
            'z_max_m': round(zmax, 2),
            'lidar_points': lidar_info['point_count'],
            'vertices_3d': verts_3d,
            'edges': edges,
        })
    
    tn = {'o': 'odkvap', 'n': 'nárožie', 'u': 'úžľabie', 'h': 'hrebeň', 'f': 'štít'}
    summary = {et: {
        'nazov': tn.get(et, et),
        'pocet': len(lengths),
        'celkom_m': round(sum(lengths), 3),
    } for et, lengths in all_edges.items()}
    
    return {'planes': planes, 'edges_summary': summary}
