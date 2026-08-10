"""Fix roof polygon expansion: extend planes toward ridge + neighbors, keep eave fixed.
Classification: narozie = angle from eave < 90°, uzlabie = angle > 90°.
"""
import numpy as np

def expand_and_classify(planes):
    """Expand polygons toward neighbors (not eave). Classify edges by angle from eave."""
    if len(planes) < 2:
        return planes
    
    # Step 1: For each plane, find its eave edges (lowest Z)
    for p in planes:
        verts = np.array(p['vertices_3d'])
        if len(verts) < 3:
            continue
        zs = verts[:, 2]
        z_min = zs.min()
        z_max = zs.max()
        
        nv = len(verts)
        for ei, e in enumerate(p.get('edges', [])):
            v1 = verts[e.get('v1', ei)]
            v2 = verts[e.get('v2', (ei+1) % nv)]
            mz = (v1[2] + v2[2]) / 2
            
            # Determine if eave: horizontal edge at lowest Z
            dz = abs(v2[2] - v1[2])
            if dz < 0.25 and abs(mz - z_min) < 0.3:
                if e['type'] not in ('h',):
                    e['type'] = 'o'
                    e['_is_eave'] = True
            elif dz < 0.25 and abs(mz - z_max) < 0.4:
                e['type'] = 'h'
                e['_is_ridge'] = True
    
    # Step 2: Classify sloped edges by angle from eave
    for p in planes:
        verts = np.array(p['vertices_3d'])
        if len(verts) < 3:
            continue
        
        nv = len(verts)
        
        # Find eave edge(s) — the ones marked
        eave_edges = []
        ridge_edges = []
        for ei, e in enumerate(p.get('edges', [])):
            if e.get('_is_eave'):
                eave_edges.append(ei)
            if e.get('_is_ridge'):
                ridge_edges.append(ei)
        
        for ei, e in enumerate(p.get('edges', [])):
            if e.get('_is_eave') or e.get('_is_ridge'):
                continue
            
            v1 = verts[e.get('v1', ei)]
            v2 = verts[e.get('v2', (ei+1) % nv)]
            edge_vec_2d = (v2 - v1)[:2]
            elen_2d = np.linalg.norm(edge_vec_2d)
            if elen_2d < 0.3:
                continue
            
            # Find closest eave edge and compute angle
            if eave_edges:
                min_angle = 180
                for eei in eave_edges:
                    ev = p['edges'][eei]
                    ev1 = verts[ev.get('v1', eei)]
                    ev2 = verts[ev.get('v2', (eei+1) % nv)]
                    eave_vec_2d = (ev2 - ev1)[:2]
                    el = np.linalg.norm(eave_vec_2d)
                    if el < 0.3:
                        continue
                    
                    # Angle between eave and this edge
                    cos_a = np.clip(np.dot(edge_vec_2d/elen_2d, eave_vec_2d/el), -1, 1)
                    angle = np.degrees(np.arccos(cos_a))
                    min_angle = min(min_angle, angle)
                
                if min_angle < 90:
                    e['type'] = 'n'  # narozie — acute angle from eave
                else:
                    e['type'] = 'u'  # uzlabie — obtuse angle from eave
            else:
                # No eave found — sloped edge = stit by default
                if e['type'] not in ('n', 'u'):
                    e['type'] = 'f'
    
    # Step 3: Expand polygons toward neighbors (extend non-eave edges)
    # For each pair of adjacent planes, extend the shared edge outward
    _expand_toward_neighbors(planes)
    
    # Clean up
    for p in planes:
        for e in p.get('edges', []):
            e.pop('_is_eave', None)
            e.pop('_is_ridge', None)
    
    return planes


def _expand_toward_neighbors(planes, expansion=2.0):
    """Expand each plane's non-eave edges outward toward neighboring planes."""
    from scipy.spatial import KDTree
    
    # Find all non-eave edge midpoints
    all_mids, all_refs = [], []
    for pi, p in enumerate(planes):
        verts = np.array(p['vertices_3d'])
        for ei, e in enumerate(p.get('edges', [])):
            if e.get('_is_eave'):
                continue
            v1 = verts[e.get('v1', ei)]
            v2 = verts[e.get('v2', (ei+1) % len(verts))]
            mid = (v1 + v2) / 2
            all_mids.append(mid)
            all_refs.append((pi, ei, p, verts, e))
    
    if len(all_mids) < 2:
        return
    
    tree = KDTree(all_mids)
    
    # Group edges by proximity → pairs from different planes
    pairs = []
    for i, (pi, ei, p, verts, e) in enumerate(all_refs):
        neighbors = tree.query_ball_point(all_mids[i], r=3.0)
        for j in neighbors:
            if j <= i:
                continue
            pj, ej, p2, verts2, e2 = all_refs[j]
            if pi == pj:
                continue
            pairs.append((pi, ei, pj, ej))
    
    # For each pair, extend both edges outward by expansion amount
    for pi, ei, pj, ej in pairs:
        e1 = planes[pi]['edges'][ei]
        e2 = planes[pj]['edges'][ej]
        
        # Compute outward direction (away from plane centroid)
        v1 = np.array(planes[pi]['vertices_3d'])
        c1 = v1.mean(axis=0)
        s1 = np.array(e1['start'])
        e1_end = np.array(e1['end'])
        mid1 = (s1 + e1_end) / 2
        dir1 = mid1 - c1
        d1n = np.linalg.norm(dir1)
        if d1n > 0.01:
            dir1 /= d1n
            s1_new = s1 + dir1[:3] * expansion
            e1_new = e1_end + dir1[:3] * expansion
        else:
            s1_new = s1
            e1_new = e1_end
        
        v2 = np.array(planes[pj]['vertices_3d'])
        c2 = v2.mean(axis=0)
        s2 = np.array(e2['start'])
        e2_end = np.array(e2['end'])
        mid2 = (s2 + e2_end) / 2
        dir2 = mid2 - c2
        d2n = np.linalg.norm(dir2)
        if d2n > 0.01:
            dir2 /= d2n
            s2_new = s2 + dir2[:3] * expansion
            e2_new = e2_end + dir2[:3] * expansion
        else:
            s2_new = s2
            e2_new = e2_end
        
        # Snap both edges to midpoint of their expanded versions
        avg_s = ((s1_new + s2_new) / 2)
        avg_e = ((e1_new + e2_new) / 2)
        
        e1['start'] = [round(float(x), 3) for x in avg_s]
        e1['end'] = [round(float(x), 3) for x in avg_e]
        e2['start'] = [round(float(x), 3) for x in avg_s]
        e2['end'] = [round(float(x), 3) for x in avg_e]
