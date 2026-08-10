#!/usr/bin/env python3
"""Trimesh Face-Normal Roof Plane Splitter.

Uses your idea: compute face normals → cluster by slope → split by connectivity.
Much more robust than DBSCAN on raw point cloud because the Delaunay mesh
already has proper face topology.
"""
import numpy as np, math, json, os, glob
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull

def mesh_to_roof_planes(ply_path, min_slope=10, max_slope=85, min_faces=50):
    """Load PLY mesh, split into roof planes by face normal + connectivity."""
    import trimesh
    mesh = trimesh.load(ply_path)
    
    # 1. Compute slope per face
    z_axis = np.array([0, 0, 1])
    dots = np.clip(np.dot(mesh.face_normals, z_axis), -1.0, 1.0)
    slopes = np.degrees(np.arccos(dots))
    
    # 2. Filter roof faces
    roof_mask = (slopes > min_slope) & (slopes < max_slope)
    roof_faces = np.where(roof_mask)[0]
    roof_normals = mesh.face_normals[roof_faces]
    
    if len(roof_faces) < min_faces:
        return []
    
    # 3. DBSCAN cluster by normal direction
    feats = StandardScaler().fit_transform(roof_normals)
    labels = DBSCAN(eps=0.15, min_samples=100).fit_predict(feats)
    
    # 4. Build face adjacency
    face_to_verts = {}
    for i in range(len(roof_faces)):
        face_to_verts[i] = set(mesh.faces[roof_faces[i]])
    
    vert_to_faces = defaultdict(set)
    for fi, verts in face_to_verts.items():
        for v in verts:
            vert_to_faces[v].add(fi)
    
    # 5. Connected components per normal cluster
    planes = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        cfaces = set(np.where(labels == cid)[0])
        visited = set()
        
        for start in cfaces:
            if start in visited:
                continue
            
            # BFS to find connected component
            stack = [start]
            comp = set()
            while stack:
                f = stack.pop()
                if f in visited or f not in cfaces:
                    continue
                visited.add(f)
                comp.add(f)
                # Find neighbors via shared vertices
                for v in face_to_verts[f]:
                    for nb in vert_to_faces[v]:
                        if nb not in visited and nb in cfaces:
                            stack.append(nb)
            
            if len(comp) < min_faces:
                continue
            
            comp_list = list(comp)
            comp_n = roof_normals[comp_list].mean(axis=0)
            comp_n /= np.linalg.norm(comp_n)
            slope = float(np.degrees(np.arccos(np.clip(np.dot(comp_n, z_axis), -1.0, 1.0))))
            
            # Get vertices of this component
            comp_verts = set()
            for f in comp_list:
                comp_verts.update(face_to_verts[f])
            comp_verts = np.array(list(comp_verts))
            
            # Project to 2D and get α-shape boundary
            verts_3d = mesh.vertices[list(comp_verts)]
            boundary_2d = _get_boundary_polygon(verts_3d[:, :2])
            
            # Fit plane equation: normal_x*x + normal_y*y + normal_z*z = d
            centroid = verts_3d.mean(axis=0)
            d = np.dot(comp_n, centroid)
            
            # Area from convex hull (in 3D)
            try:
                hull = ConvexHull(verts_3d[:, :2])
                area_2d = float(hull.volume)
                area_3d = area_2d / max(abs(comp_n[2]), 0.01)
            except:
                area_2d = area_3d = 0
            
            # Per-vertex Z from plane equation: z = (d - n_x*x - n_y*y) / n_z
            def plane_z(x, y):
                if abs(comp_n[2]) > 1e-6:
                    return (d - comp_n[0]*x - comp_n[1]*y) / comp_n[2]
                return centroid[2]
            
            # Compute edge lengths from boundary polygon with real 3D Z
            edges_3d = []
            bv = boundary_2d
            for ei in range(len(bv)):
                z1 = float(plane_z(bv[ei][0], bv[ei][1]))
                z2 = float(plane_z(bv[(ei+1)%len(bv)][0], bv[(ei+1)%len(bv)][1]))
                v1 = np.array([bv[ei][0], bv[ei][1], z1])
                v2 = np.array([bv[(ei+1)%len(bv)][0], bv[(ei+1)%len(bv)][1], z2])
                elen = float(np.linalg.norm(v2 - v1))
                if elen < 0.3:
                    continue
                etype = 'o'  # Default; edges with varying Z are 'f' (stit)
                if abs(z2 - z1) > 0.3: etype = 'f'
                edges_3d.append({
                    'id': f'e{len(edges_3d)+1}',
                    'type': etype,
                    'length_m': round(elen, 3),
                    'v1': ei,
                    'v2': (ei+1) % len(bv),
                    'start': [round(float(bv[ei][0]),3), round(float(bv[ei][1]),3), round(z1, 3)],
                    'end': [round(float(bv[(ei+1)%len(bv)][0]),3), round(float(bv[(ei+1)%len(bv)][1]),3), round(z2, 3)],
                })
            
            zs = verts_3d[:, 2]
            planes.append({
                'id': f'R{len(planes)+1}',
                'type': 'sedlova' if 15 <= slope <= 45 else 'valbova',
                'area_m2': round(float(area_3d), 2),
                'pitch_deg': round(slope, 1),
                'z_min_m': round(float(zs.min()), 2),
                'z_max_m': round(float(zs.max()), 2),
                'n_faces': len(comp),
                'vertices_3d': [[round(float(bv[i][0]),3), round(float(bv[i][1]),3), round(float(plane_z(bv[i][0], bv[i][1])),3)] for i in range(len(bv))],
                'edges': edges_3d,
            })
    
    return planes

def _get_boundary_polygon(points_2d):
    """Get simplified boundary polygon from 2D points using α-shape or convex hull."""
    if len(points_2d) < 4:
        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except:
            return points_2d
    try:
        import alphashape
        alpha = 2.0
        shape = alphashape.alphashape(points_2d, alpha)
        if shape.geom_type == 'Polygon':
            return np.array(shape.exterior.coords)[:-1]
        elif shape.geom_type == 'MultiPolygon':
            largest = max(shape.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords)[:-1]
    except:
        pass
    try:
        hull = ConvexHull(points_2d)
        return points_2d[hull.vertices]
    except:
        return points_2d
