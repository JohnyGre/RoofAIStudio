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
ANGLE_TOL = 30.0     # degrees for colinear merge
EDGE_NEAR = 0.5     # m - edges closer than this are neighbors

def _simplify_edge_points(points_3d, max_verts=8):
    """DP simplification to get max_verts key corners."""
    pts = np.asarray(points_3d)
    if len(pts) <= max_verts:
        return pts
    import cv2
    contour = pts[:, :2].astype(np.float32).reshape(-1, 1, 2)
    diag = np.linalg.norm(pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0))
    eps_lo, eps_hi = diag * 0.01, diag * 0.50
    for _ in range(12):
        eps = (eps_lo + eps_hi) / 2
        simp = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if len(simp) <= max_verts:
            eps_hi = eps
        else:
            eps_lo = eps
    simp = cv2.approxPolyDP(contour, eps_hi, True).reshape(-1, 2)
    result = []
    for s in simp:
        dists = np.linalg.norm(pts[:, :2] - s, axis=1)
        result.append(pts[np.argmin(dists)])
    return np.array(result)

def _fix_roof_topology(planes, snap_distance=0.6, z_tolerance=0.3):
    """Snap nearby 'f' edges between planes -> narozie/uzlabie. Fix false 'o' near ridge."""
    from scipy.spatial import KDTree
    
    edge_centroids = []
    edge_refs = []
    for p_idx, plane in enumerate(planes):
        for e_idx, edge in enumerate(plane.get("edges", [])):
            p1 = np.array(edge["start"])
            p2 = np.array(edge["end"])
            centroid = (p1 + p2) / 2.0
            edge_centroids.append(centroid)
            edge_refs.append({"p_idx": p_idx, "e_idx": e_idx, "type": edge["type"]})
    
    if not edge_centroids:
        return planes
    
    tree = KDTree(edge_centroids)
    processed_pairs = set()
    
    for i, ref in enumerate(edge_refs):
        if ref["type"] != "f":
            continue
        neighbors = tree.query_ball_point(edge_centroids[i], r=snap_distance)
        
        for n_idx in neighbors:
            if n_idx == i or tuple(sorted([i, n_idx])) in processed_pairs:
                continue
            neighbor_ref = edge_refs[n_idx]
            if ref["p_idx"] != neighbor_ref["p_idx"] and neighbor_ref["type"] in ["f", "o"]:
                processed_pairs.add(tuple(sorted([i, n_idx])))
                
                plane1 = planes[ref["p_idx"]]
                plane2 = planes[neighbor_ref["p_idx"]]
                
                c1 = np.mean([v for v in plane1.get("vertices_3d", [])], axis=0)
                c2 = np.mean([v for v in plane2.get("vertices_3d", [])], axis=0)
                
                edge_z = edge_centroids[i][2]
                avg_plane_z = (c1[2] + c2[2]) / 2.0
                new_type = "n" if edge_z > avg_plane_z else "u"
                
                planes[ref["p_idx"]]["edges"][ref["e_idx"]]["type"] = new_type
                planes[neighbor_ref["p_idx"]]["edges"][neighbor_ref["e_idx"]]["type"] = new_type
                
                orig_edge1 = planes[ref["p_idx"]]["edges"][ref["e_idx"]]
                orig_edge2 = planes[neighbor_ref["p_idx"]]["edges"][neighbor_ref["e_idx"]]
                
                avg_start = (np.array(orig_edge1["start"]) + np.array(orig_edge2["start"])) / 2.0
                avg_end = (np.array(orig_edge1["end"]) + np.array(orig_edge2["end"])) / 2.0
                
                orig_edge1["start"] = avg_start.tolist()
                orig_edge1["end"] = avg_end.tolist()
                orig_edge2["start"] = avg_start.tolist()
                orig_edge2["end"] = avg_end.tolist()
    
    # Classify sloped edges by angle from CONNECTED eave (not any eave)
    for p in planes:
        verts = np.array(p.get('vertices_3d', []))
        if len(verts) < 3:
            continue
        nv = len(verts)
        z_min = min(v[2] for v in verts)
        
        # Mark edges
        for ei, e in enumerate(p.get('edges', [])):
            v1 = verts[e.get('v1', ei)]
            v2 = verts[e.get('v2', (ei+1) % nv)]
            mz = (v1[2] + v2[2]) / 2
            dz = abs(v2[2] - v1[2])
            e['_v1'] = v1; e['_v2'] = v2
            e['_is_eave'] = (dz < 0.3 and abs(mz - z_min) < 0.4)
            e['_is_ridge'] = (dz < 0.3 and not e['_is_eave'])
        
        for ei, e in enumerate(p.get('edges', [])):
            if e.get('_is_eave') or e.get('_is_ridge'):
                continue
            if e['type'] in ('n', 'u'):
                continue  # already classified
            
            ed = (e['_v2'] - e['_v1'])[:2]
            en = np.linalg.norm(ed)
            if en < 0.3: continue
            ed_u = ed / en
            
            # Find connected eave edge (shares a vertex)
            v1_idx = e.get('v1', ei)
            v2_idx = e.get('v2', (ei+1) % nv)
            
            angle = 90
            for ej, e2 in enumerate(p.get('edges', [])):
                if not e2.get('_is_eave'): continue
                ev1 = e2.get('v1', ej)
                ev2 = e2.get('v2', (ej+1) % nv)
                # Check shared vertex
                if v1_idx == ev2 or v2_idx == ev1:
                    # eave enters at start, edge leaves at start
                    eav = (e2['_v1'] - e2['_v2'])[:2] if v1_idx == ev2 else (e2['_v2'] - e2['_v1'])[:2]
                elif v1_idx == ev1 or v2_idx == ev2:
                    eav = (e2['_v2'] - e2['_v1'])[:2] if v1_idx == ev1 else (e2['_v1'] - e2['_v2'])[:2]
                else:
                    continue
                en_ = np.linalg.norm(eav)
                if en_ < 0.3: continue
                cos_a = np.dot(ed_u, eav / en_)
                angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                break
            
            if angle < 90:
                e['type'] = 'n'
            elif angle > 90:
                e['type'] = 'u'
    
    # Clean up
    for p in planes:
        for e in p.get('edges', []):
            e.pop('_v1', None); e.pop('_v2', None)
        for k in ('start', 'end'):
            if k in e and hasattr(e.get(k), 'tolist'):
                e[k] = list(e[k])
            e.pop('_is_eave', None); e.pop('_is_ridge', None)
    
    # Fix false     # Fix false 'o' edges near ridge
    for plane in planes:
        verts = plane.get("vertices_3d", [])
        if not verts:
            continue
        z_coords = [v[2] for v in verts]
        max_z = np.percentile(z_coords, 90)  # 90th percentile, not absolute max
        
        for edge in plane.get("edges", []):
            if edge["type"] == "o":
                avg_edge_z = (edge["start"][2] + edge["end"][2]) / 2.0
                if abs(max_z - avg_edge_z) < z_tolerance:
                    edge["type"] = "h"
    
    return planes


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
    
    # Classify edges with inter-plane neighbor context
    _classify_edges(planes)
    
    # Fix topology + expand + reclassify by angle from eave
    _fix_roof_topology(planes, snap_distance=0.8)

    
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
        epsilon = diag * 0.03
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
    
    # Classify edges with inter-plane neighbor context
    _classify_edges(planes)
    
    # Fix topology + expand + reclassify by angle from eave
    _fix_roof_topology(planes, snap_distance=0.8)

    
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
