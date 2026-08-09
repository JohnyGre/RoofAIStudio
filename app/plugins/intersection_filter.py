"""Filter and merge intersection edges for clean output."""
import numpy as np

def filter_intersections(inter_list, planes, min_pitch=8, min_len=1.0, merge_tol=0.5):
    """Filter intersection edges: remove false positives, merge nearby, classify properly.
    
    Args:
        inter_list: raw intersections from compute_all_intersections
        planes: list of plane dicts with 'pitch'
        min_pitch: minimum pitch for both planes
        min_len: minimum edge length in meters
        merge_tol: distance tolerance for merging parallel edges
    
    Returns filtered list.
    """
    filtered = []
    
    for e in inter_list:
        pi, pj = e['planes']
        # Both planes must have meaningful pitch
        if planes[pi]['pitch'] < min_pitch or planes[pj]['pitch'] < min_pitch:
            continue
        
        if e['length_m'] < min_len:
            continue
        
        # Reclassify: determine convex/concave by edge Z vs plane centroid Z
        n1, n2 = planes[pi]['n'], planes[pj]['n']
        
        edge_dir = np.array(e['end']) - np.array(e['start'])
        edge_dir /= np.linalg.norm(edge_dir) + 1e-10
        
        # Convex if edge is ABOVE both plane centroids (ridge meeting at top)
        # Concave if edge is BELOW either plane centroid (valley meeting at bottom)
        edge_mid_z = (e['start'][2] + e['end'][2]) / 2
        z1 = planes[pi].get('z_mean', planes[pi].get('c', 0))
        z2 = planes[pj].get('z_mean', planes[pj].get('c', 0))
        
        # Edge at highest Z of the two planes = convex
        # Edge at lowest Z = valley
        is_convex = edge_mid_z > max(z1, z2) * 0.8
        
        # Edge direction horizontality
        horiz = np.linalg.norm(edge_dir[:2])
        slope_ratio = abs(edge_dir[2]) / max(horiz, 0.001)
        
        if is_convex:
            etype = 'h' if slope_ratio < 0.25 else 'n'
        else:
            etype = 'u'
        
        e['type'] = etype
        
        filtered.append(e)
    
    # Merge nearly parallel, collinear edges
    merged = _merge_edges(filtered, merge_tol)
    
    return merged

def _merge_edges(edges, tol):
    """Merge edges that are on the same line and close."""
    if len(edges) <= 1:
        return edges
    
    # Sort by direction
    used = [False] * len(edges)
    merged = []
    
    for i in range(len(edges)):
        if used[i]: continue
        group = [edges[i]]
        used[i] = True
        
        dir_i = np.array(edges[i]['end']) - np.array(edges[i]['start'])
        dir_i /= np.linalg.norm(dir_i) + 1e-10
        
        for j in range(i+1, len(edges)):
            if used[j]: continue
            if edges[j]['type'] != edges[i]['type']: continue
            
            dir_j = np.array(edges[j]['end']) - np.array(edges[j]['start'])
            dir_j /= np.linalg.norm(dir_j) + 1e-10
            
            # Check if parallel
            if abs(np.dot(dir_i, dir_j)) < 0.9:
                continue
            
            # Check if on same line (distance from first edge's line)
            mid_j = (np.array(edges[j]['start']) + np.array(edges[j]['end'])) / 2
            dist = np.linalg.norm(np.cross(mid_j - np.array(edges[i]['start']), dir_i))
            
            if dist < tol:
                group.append(edges[j])
                used[j] = True
        
        if len(group) > 1:
            # Merge: extend the longest edge to cover all projections
            all_pts = []
            for e in group:
                all_pts.append(np.array(e['start']))
                all_pts.append(np.array(e['end']))
            
            # Project all points onto the first edge's line
            origin = all_pts[0]
            t_vals = [np.dot(p - origin, dir_i) for p in all_pts]
            t_min, t_max = min(t_vals), max(t_vals)
            
            merged.append({
                'planes': edges[i]['planes'],
                'type': edges[i]['type'],
                'length_m': round(float(t_max - t_min), 3),
                'start': (origin + dir_i * t_min).tolist(),
                'end': (origin + dir_i * t_max).tolist(),
            })
        else:
            merged.append(edges[i])
    
    return merged
