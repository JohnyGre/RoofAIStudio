#!/usr/bin/env python3
"""
Practical roof measurement from point cloud — convex hull edges + material calculator.
"""
import numpy as np, math, os
from scipy.spatial import ConvexHull, Delaunay
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

def analyze_roof_from_points(points, ground_z):
    """
    Analyze building points and extract:
    - Convex hull perimeter (eave length ≈ gutter + fascia)
    - Individual roof planes with area + pitch
    - Ridge/valley/hip estimates
    Returns dict with all measurements + material quantities.
    """
    pts = points.copy()
    # Center
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    pts[:,0] -= cx; pts[:,1] -= cy
    pts[:,2] -= ground_z
    
    result = {'ground_z': round(float(ground_z), 2)}
    
    # 1. Convex hull → perimeter (eave estimate)
    hull_2d = ConvexHull(pts[:, :2])
    result['eave_perimeter_m'] = round(float(hull_2d.area), 2)  # In 2D hull.area = perimeter
    result['footprint_area_m2'] = round(float(hull_2d.volume), 1)  # In 2D hull.volume = area
    
    # Get hull boundary as ordered edge list
    hull_verts = pts[hull_2d.vertices, :2]
    result['boundary_points'] = len(hull_2d.vertices)
    
    # 2. Find roof planes via normal clustering
    # Downsample for speed
    n_pts = len(pts)
    sample = pts if n_pts <= 5000 else pts[np.random.choice(n_pts, 5000, replace=False)]
    
    # Compute local normals via PCA
    tree = KDTree(sample[:, :2])
    normals = np.zeros((len(sample), 3))
    for i in range(len(sample)):
        idx = tree.query_radius(sample[i:i+1, :2], r=1.5)[0]
        if len(idx) >= 8:
            pca = PCA(n_components=3).fit(sample[idx])
            n = pca.components_[2]
            if n[2] < 0: n = -n
            normals[i] = n
        else:
            normals[i] = [0, 0, 1]
    
    # Cluster by normal direction (ignore flat/ground normals)
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    
    features = np.column_stack([normals, sample[:, 2:]])
    scaler = StandardScaler()
    labels = DBSCAN(eps=0.35, min_samples=15).fit_predict(scaler.fit_transform(features))
    
    roof_planes = []
    for cid in sorted(set(labels)):
        if cid == -1: continue
        mask = labels == cid
        cluster_pts = sample[mask]
        avg_normal = normals[mask].mean(axis=0)
        avg_normal /= np.linalg.norm(avg_normal)
        pitch = math.degrees(math.acos(abs(avg_normal[2])))
        
        # Skip flat planes (< 5° pitch) — they're ground/noise
        if pitch < 5:
            continue
        
        # Filter full point cloud by proximity to this cluster's XY extent
        cx_c = cluster_pts[:, 0].mean()
        cy_c = cluster_pts[:, 1].mean()
        rx = float(np.max(cluster_pts[:, 0]) - np.min(cluster_pts[:, 0])) / 2 + 1
        ry = float(np.max(cluster_pts[:, 1]) - np.min(cluster_pts[:, 1])) / 2 + 1
        
        # Get all points near this cluster
        nearby = pts[(np.abs(pts[:, 0] - cx_c) < rx) & (np.abs(pts[:, 1] - cy_c) < ry)]
        
        if len(nearby) < 30:
            continue
        
        # Fit plane to nearby points
        pca2 = PCA(n_components=3).fit(nearby)
        plane_normal = pca2.components_[2]
        if plane_normal[2] < 0: plane_normal = -plane_normal
        pitch2 = math.degrees(math.acos(abs(plane_normal[2])))
        
        # Project points to 2D for area calculation
        # Create orthonormal basis for the plane
        z_axis = np.array([0, 0, 1])
        u = np.cross(plane_normal, z_axis)
        u /= np.linalg.norm(u) if np.linalg.norm(u) > 1e-10 else 1
        v = np.cross(plane_normal, u)
        v /= np.linalg.norm(v)
        
        # Project points
        proj_u = np.dot(nearby[:, :2], u[:2])
        proj_v = np.dot(nearby[:, :2], v[:2])
        
        try:
            hull_plane = ConvexHull(np.column_stack([proj_u, proj_v]))
            area_3d = float(hull_plane.volume)
        except:
            area_3d = ((np.max(proj_u) - np.min(proj_u)) * (np.max(proj_v) - np.min(proj_v)))
        
        if area_3d < 2:
            continue
        
        z_mean = float(nearby[:, 2].mean())
        z_max = float(nearby[:, 2].max())
        z_min = float(nearby[:, 2].min())
        
        roof_planes.append({
            'area_3d_m2': round(float(area_3d), 2),
            'pitch_deg': round(float(pitch2), 1),
            'normal': [round(float(plane_normal[0]), 3), round(float(plane_normal[1]), 3), round(float(plane_normal[2]), 3)],
            'z_mean_m': round(z_mean, 2),
            'z_max_m': round(z_max, 2),
            'z_min_m': round(z_min, 2),
            'point_count': len(nearby),
            'center': [round(float(cx_c), 2), round(float(cy_c), 2)],
        })
    
    result['roof_planes'] = roof_planes
    result['total_roof_area_3d_m2'] = round(sum(p['area_3d_m2'] for p in roof_planes), 2)
    
    # 3. Edge classification between planes
    ridges, hips, valleys = [], [], []
    for i in range(len(roof_planes)):
        for j in range(i+1, len(roof_planes)):
            n1 = np.array(roof_planes[i]['normal'])
            n2 = np.array(roof_planes[j]['normal'])
            
            # Check if planes intersect
            cross = np.cross(n1, n2)
            cross_len = np.linalg.norm(cross)
            if cross_len < 0.05:
                continue  # Parallel
            
            # Edge direction
            edge_dir = cross / cross_len
            slope = abs(edge_dir[2]) / max(np.linalg.norm(edge_dir[:2]), 0.001)
            
            # Determine ridge/hip/valley
            mid_n = (n1 + n2) / 2
            is_convex = mid_n[2] > 0
            z_diff = abs(roof_planes[i]['z_mean_m'] - roof_planes[j]['z_mean_m'])
            
            # Estimate length: intersection of bounding boxes
            c1 = np.array(roof_planes[i]['center'])
            c2 = np.array(roof_planes[j]['center'])
            dist = np.linalg.norm(c1 - c2)
            est_len = min(5.0, dist)  # Rough estimate
            
            edge = {
                'planes': [i, j],
                'direction': [round(float(edge_dir[0]), 3), round(float(edge_dir[1]), 3), round(float(edge_dir[2]), 3)],
                'length_est_m': round(float(est_len), 2),
            }
            
            if is_convex:
                if slope < 0.3:
                    ridges.append(edge)
                else:
                    hips.append(edge)
            else:
                valleys.append(edge)
    
    result['ridges'] = ridges
    result['hips'] = hips
    result['valleys'] = valleys
    result['ridge_total_m'] = round(sum(e['length_est_m'] for e in ridges), 2)
    result['hip_total_m'] = round(sum(e['length_est_m'] for e in hips), 2)
    result['valley_total_m'] = round(sum(e['length_est_m'] for e in valleys), 2)
    
    # 4. Material calculator
    total_roof = result['total_roof_area_3d_m2']
    waste_tile = 0.10  # 10% waste for tiles
    waste_foil = 0.15  # 15% for underlayment
    
    result['materials'] = {
        'skridla_m2': round(total_roof * (1 + waste_tile), 1),
        'folia_m2': round(total_roof * (1 + waste_foil), 1),
        'latovanie_m2': round(total_roof, 1),
        'okapove_plechy_m': round(result['eave_perimeter_m'] * 1.05, 2),  # +5% waste
        'zlab_m': round(result['eave_perimeter_m'] * 1.02, 2),
        'hrebenace_m': round(result['ridge_total_m'] + result['hip_total_m'], 2),
        'uzlabie_m': round(result['valley_total_m'] * 1.05, 2),
    }
    
    return result
