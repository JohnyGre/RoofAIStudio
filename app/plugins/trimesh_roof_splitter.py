#!/usr/bin/env python3
"""Trimesh Face-Normal Roof Plane Splitter v2.

Face normals → DBSCAN slope clusters → connected components → 
edge simplification → edge classification (odkvap/hreben/stit/narozie/uzlabie).
"""
import numpy as np, math, json, os, glob
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull

Z_TOL = 0.25        # tolerance for "horizontal" edge (m)
ANGLE_TOL = 15.0     # degrees for colinear merge
EDGE_NEAR = 0.5     # m - edges closer than this are neighbors

def _simplify_edge_points(points_3d, angle_tolerance=ANGLE_TOL):
    """Merge colinear edge points; keep only real corners."""
    pts = np.asarray(points_3d)
    if len(pts) < 3:
        return pts
    simplified = [pts[0]]
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i-1]
        v2 = pts[i+1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cos_a = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        if angle > angle_tolerance:
            simplified.append(pts[i])
    simplified.append(pts[-1])
    return np.array(simplified)

def _classify_edges(planes):
    """Classify every edge across all planes using Z-context + neighbor normals.
    
    Modifies planes in-place: replaces edges with classified versions,
    adds vertices_3d from simplified polygon.
    """
    # Collect all edges with their plane index and normal
    all_edges = []  # (pi, ei_idx, start_3d, end_3d, length)
    for pi, p in enumerate(planes):
        verts = np.array(p.get('_raw_vertices', p.get('vertices_3d', [])))
        if len(verts) < 3:
            continue
        # Simplify
        if len(verts) > 6:
            verts = _simplify_edge_points(verts)
        if len(verts) < 3:
            verts = np.array(p.get('_raw_vertices', p.get('vertices_3d', [])))
        
        # Build edges
        edges = []
        n = len(verts)
        for ei in range(n):
            v1 = verts[ei]
            v2 = verts[(ei+1) % n]
            elen = float(np.linalg.norm(v2 - v1))
            if elen < 0.5:
                continue
            edges.append({'start': v1, 'end': v2, 'length': elen, 'idx': ei})
            all_edges.append((pi, len(edges)-1, v1, v2, elen))
        
        p['_verts_simplified'] = verts
        p['_raw_edges'] = edges
    
    # Precompute neighbor map: which plane pairs share an edge
    # Two edges are neighbors if they're in different planes and within EDGE_NEAR
    edge_neighbors = defaultdict(list)  # (pi, ei) -> [(pj, ej, pj_normal)]
    for i in range(len(all_edges)):
        pi, ei, s1, e1, l1 = all_edges[i]
        mid1 = (s1 + e1) / 2
        for j in range(i+1, len(all_edges)):
            pj, ej, s2, e2, l2 = all_edges[j]
            if pi == pj:
                continue
            mid2 = (s2 + e2) / 2
            # Check midpoint proximity
            if np.linalg.norm(mid1 - mid2) < EDGE_NEAR:
                edge_neighbors[(pi, ei)].append((pj, planes[pj].get('_normal')))
                edge_neighbors[(pj, ej)].append((pi, planes[pi].get('_normal')))
    
    # Classify each plane's edges
    for pi, p in enumerate(planes):
        verts = np.array(p.get('_verts_simplified', p.get('vertices_3d', [])))
        if len(verts) < 3:
            p['edges'] = p.get('_raw_edges', [])
            continue
        
        n = len(verts)
        z_min = float(verts[:, 2].min())
        z_max = float(verts[:, 2].max())
        plane_n = p.get('_normal', np.array([0,0,1]))
        
        classified = []
        for ei in range(n):
            v1 = verts[ei]
            v2 = verts[(ei+1) % n]
            elen = float(np.linalg.norm(v2 - v1))
            if elen < 0.5:
                continue
            
            dz = abs(v2[2] - v1[2])
            mz = (v1[2] + v2[2]) / 2
            
            # Find neighbors for this edge
            neighbors = edge_neighbors.get((pi, ei), [])
            neighbor_normals = [n for _, n in neighbors]
            
            # Classification
            if dz <= Z_TOL:
                # Horizontal edge
                if abs(mz - z_max) < Z_TOL * 2:
                    etype = 'h'       # hreben
                elif abs(mz - z_min) < Z_TOL * 2:
                    etype = 'o'       # odkvap
                else:
                    etype = 'o'       # intermediate horizontal → odkvap
            else:
                # Sloped edge
                if not neighbor_normals:
                    etype = 'f'       # stit (no neighbor)
                else:
                    # Determine convex/concave from cross product
                    n1 = plane_n / np.linalg.norm(plane_n)
                    n2 = neighbor_normals[0] / np.linalg.norm(neighbor_normals[0])
                    cross_p = np.cross(n1, n2)
                    edge_v = (v2 - v1) / elen
                    convexity = np.dot(cross_p, edge_v)
                    if convexity > 0.01:
                        etype = 'n'   # narozie
                    elif convexity < -0.01:
                        etype = 'u'   # uzlabie
                    else:
                        etype = 'f'   # stit
            
            classified.append({
                'id': '{}{}'.format(etype, len(classified)+1),
                'type': etype,
                'length_m': round(elen, 3),
                'v1': ei,
                'v2': (ei+1) % n,
                'start': [round(float(v1[0]),3), round(float(v1[1]),3), round(float(v1[2]),3)],
                'end':   [round(float(v2[0]),3), round(float(v2[1]),3), round(float(v2[2]),3)],
            })
        
        p['edges'] = classified
        p['vertices_3d'] = [[round(float(v[0]),3), round(float(v[1]),3), round(float(v[2]),3)] for v in verts]


def mesh_to_roof_planes(ply_path, min_slope=10, max_slope=85, min_faces=50):
    """Load PLY mesh, split into roof planes by face normal + connectivity."""
    import trimesh
    mesh = trimesh.load(ply_path)
    z_axis = np.array([0, 0, 1])
    dots = np.clip(np.dot(mesh.face_normals, z_axis), -1.0, 1.0)
    slopes = np.degrees(np.arccos(dots))
    
    roof_mask = (slopes > min_slope) & (slopes < max_slope)
    roof_faces = np.where(roof_mask)[0]
    roof_normals = mesh.face_normals[roof_faces]
    
    if len(roof_faces) < min_faces:
        return []
    
    feats = StandardScaler().fit_transform(roof_normals)
    labels = DBSCAN(eps=0.15, min_samples=100).fit_predict(feats)
    
    face_to_verts = {i: set(mesh.faces[roof_faces[i]]) for i in range(len(roof_faces))}
    vert_to_faces = defaultdict(set)
    for fi, verts in face_to_verts.items():
        for v in verts:
            vert_to_faces[v].add(fi)
    
    planes = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        cfaces = set(np.where(labels == cid)[0])
        visited = set()
        
        for start in cfaces:
            if start in visited:
                continue
            stack = [start]; comp = set()
            while stack:
                f = stack.pop()
                if f in visited or f not in cfaces:
                    continue
                visited.add(f); comp.add(f)
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
            
            comp_verts_set = set()
            for f in comp_list:
                comp_verts_set.update(face_to_verts[f])
            verts_3d = mesh.vertices[list(comp_verts_set)]
            
            boundary_2d = _get_boundary_polygon(verts_3d[:, :2])
            
            centroid = verts_3d.mean(axis=0)
            d = np.dot(comp_n, centroid)
            def plane_z(x, y):
                if abs(comp_n[2]) > 1e-6:
                    return (d - comp_n[0]*x - comp_n[1]*y) / comp_n[2]
                return centroid[2]
            
            # Raw 3D boundary vertices (before simplification)
            raw_verts_3d = [[round(float(boundary_2d[i][0]),3), round(float(boundary_2d[i][1]),3), round(float(plane_z(boundary_2d[i][0], boundary_2d[i][1])),3)] for i in range(len(boundary_2d))]
            
            try:
                hull = ConvexHull(verts_3d[:, :2])
                area_2d = float(hull.volume)
                area_3d = area_2d / max(abs(comp_n[2]), 0.01)
            except:
                area_2d = area_3d = 0
            
            zs = verts_3d[:, 2]
            planes.append({
                'id': f'R{len(planes)+1}',
                'type': 'sedlova' if 15 <= slope <= 45 else 'valbova',
                'area_m2': round(float(area_3d), 2),
                'pitch_deg': round(slope, 1),
                'z_min_m': round(float(zs.min()), 2),
                'z_max_m': round(float(zs.max()), 2),
                'n_faces': len(comp),
                '_normal': comp_n,
                '_raw_vertices': raw_verts_3d,
                'vertices_3d': raw_verts_3d,  # will be overwritten by classify
                'edges': [],  # will be filled by classify
            })
    
    # Classify all edges with inter-plane neighbor context
    _classify_edges(planes)
    
    # Clean up internal fields
    for p in planes:
        p.pop('_normal', None)
        p.pop('_raw_vertices', None)
        p.pop('_verts_simplified', None)
        p.pop('_raw_edges', None)
    
    return planes


def _get_boundary_polygon(points_2d):
    """Get boundary polygon from 2D points using ConvexHull + DP simplification."""
    if len(points_2d) < 4:
        return points_2d
    try:
        hull = ConvexHull(points_2d)
        hull_pts = points_2d[hull.vertices]
        import cv2
        contour = np.array(hull_pts, dtype=np.float32).reshape(-1, 1, 2)
        diag = np.linalg.norm(points_2d.max(axis=0) - points_2d.min(axis=0))
        epsilon = diag * 0.02
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        result = simplified.reshape(-1, 2)
        if len(result) >= 3:
            return result
        return hull_pts
    except:
        return points_2d

def mesh_to_roof_planes(ply_path, min_slope=10, max_slope=85, min_faces=50):
    """Load PLY mesh, split into roof planes by face normal + connectivity."""
    import trimesh
    mesh = trimesh.load(ply_path)
    z_axis = np.array([0, 0, 1])
    dots = np.clip(np.dot(mesh.face_normals, z_axis), -1.0, 1.0)
    slopes = np.degrees(np.arccos(dots))
    
    roof_mask = (slopes > min_slope) & (slopes < max_slope)
    roof_faces = np.where(roof_mask)[0]
    roof_normals = mesh.face_normals[roof_faces]
    
    if len(roof_faces) < min_faces:
        return []
    
    feats = StandardScaler().fit_transform(roof_normals)
    labels = DBSCAN(eps=0.15, min_samples=100).fit_predict(feats)
    
    face_to_verts = {i: set(mesh.faces[roof_faces[i]]) for i in range(len(roof_faces))}
    vert_to_faces = defaultdict(set)
    for fi, verts in face_to_verts.items():
        for v in verts:
            vert_to_faces[v].add(fi)
    
    planes = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        cfaces = set(np.where(labels == cid)[0])
        visited = set()
        
        for start in cfaces:
            if start in visited:
                continue
            stack = [start]; comp = set()
            while stack:
                f = stack.pop()
                if f in visited or f not in cfaces:
                    continue
                visited.add(f); comp.add(f)
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
            
            comp_verts_set = set()
            for f in comp_list:
                comp_verts_set.update(face_to_verts[f])
            verts_3d = mesh.vertices[list(comp_verts_set)]
            
            boundary_2d = _get_boundary_polygon(verts_3d[:, :2])
            
            centroid = verts_3d.mean(axis=0)
            d = np.dot(comp_n, centroid)
            def plane_z(x, y):
                if abs(comp_n[2]) > 1e-6:
                    return (d - comp_n[0]*x - comp_n[1]*y) / comp_n[2]
                return centroid[2]
            
            # Raw 3D boundary vertices (before simplification)
            raw_verts_3d = [[round(float(boundary_2d[i][0]),3), round(float(boundary_2d[i][1]),3), round(float(plane_z(boundary_2d[i][0], boundary_2d[i][1])),3)] for i in range(len(boundary_2d))]
            
            try:
                hull = ConvexHull(verts_3d[:, :2])
                area_2d = float(hull.volume)
                area_3d = area_2d / max(abs(comp_n[2]), 0.01)
            except:
                area_2d = area_3d = 0
            
            zs = verts_3d[:, 2]
            planes.append({
                'id': f'R{len(planes)+1}',
                'type': 'sedlova' if 15 <= slope <= 45 else 'valbova',
                'area_m2': round(float(area_3d), 2),
                'pitch_deg': round(slope, 1),
                'z_min_m': round(float(zs.min()), 2),
                'z_max_m': round(float(zs.max()), 2),
                'n_faces': len(comp),
                '_normal': comp_n,
                '_raw_vertices': raw_verts_3d,
                'vertices_3d': raw_verts_3d,  # will be overwritten by classify
                'edges': [],  # will be filled by classify
            })
    
    # Classify all edges with inter-plane neighbor context
    _classify_edges(planes)
    
    # Clean up internal fields
    for p in planes:
        p.pop('_normal', None)
        p.pop('_raw_vertices', None)
        p.pop('_verts_simplified', None)
        p.pop('_raw_edges', None)
    
    return planes


def _get_boundary_polygon(points_2d):
    """Get boundary polygon from 2D points using α-shape or convex hull."""
    if len(points_2d) < 4:
        try:
            hull = ConvexHull(points_2d)
            return points_2d[hull.vertices]
        except:
            return points_2d
    try:
        import alphashape
        alpha = 5.0
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
