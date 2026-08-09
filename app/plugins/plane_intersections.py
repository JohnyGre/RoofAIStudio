"""Compute real intersection edges between roof planes to classify n/u/h edges."""
import numpy as np, math
from scipy.spatial import ConvexHull

def plane_intersection_line(n1, c1, n2, c2):
    """Compute intersection line of two planes: n1·x = d1, n2·x = d2.
    Returns (origin, direction) or None if parallel."""
    cross = np.cross(n1, n2)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 0.02:
        return None  # Nearly parallel
    direction = cross / cross_norm
    
    # Find a point: solve n1·p = d1, n2·p = d2, direction·p = 0
    A = np.array([n1, n2, direction])
    d1_val = float(c1 * n1[2])  # z = a*x + b*y + c → n1·(x,y,z) = c1 * n1_z
    d2_val = float(c2 * n2[2])
    b = np.array([d1_val, d2_val, 0])
    try:
        origin = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return origin, direction

def clip_line_to_polygon_2d(origin, direction, poly_2d, plane_basis_u, plane_basis_v):
    """Clip an infinite line to a 2D polygon, returning (start_pt, end_pt) in 3D or None."""
    if len(poly_2d) < 3:
        return None
    
    # Convert line to 2D in polygon's coordinate system
    # Project origin and direction to 2D
    o2d = np.array([np.dot(origin[:2], plane_basis_u[:2]),
                     np.dot(origin[:2], plane_basis_v[:2])])
    d2d = np.array([np.dot(direction[:2], plane_basis_u[:2]),
                     np.dot(direction[:2], plane_basis_v[:2])])
    
    # Find intersections with each polygon edge
    t_values = []
    n = len(poly_2d)
    for i in range(n):
        p1 = poly_2d[i]
        p2 = poly_2d[(i+1) % n]
        edge = p2 - p1
        # Solve: o2d + t*d2d = p1 + s*edge
        denom = d2d[0]*edge[1] - d2d[1]*edge[0]
        if abs(denom) < 1e-10:
            continue
        t = ((p1[0] - o2d[0])*edge[1] - (p1[1] - o2d[1])*edge[0]) / denom
        s = ((p1[0] - o2d[0])*d2d[1] - (p1[1] - o2d[1])*d2d[0]) / denom
        if 0 <= s <= 1:
            t_values.append(t)
    
    t_values.sort()
    if len(t_values) < 2:
        return None
    
    t_start, t_end = t_values[0], t_values[-1]
    
    # Convert back to 3D
    start_3d = origin + direction * t_start
    end_3d = origin + direction * t_end
    
    return start_3d, end_3d

def compute_all_intersections(planes):
    """For each plane pair, compute intersection edge and classify.
    planes: list of dicts with 'n', 'c', 'u', 'v', 'hull2d', 'pitch', 'z_max', 'z_min'
    Returns list of intersection edges with type assignment."""
    
    intersections = []
    
    for i in range(len(planes)):
        for j in range(i+1, len(planes)):
            p1, p2 = planes[i], planes[j]
            
            # Get intersection line
            result = plane_intersection_line(p1['n'], p1['c'], p2['n'], p2['c'])
            if result is None:
                continue
            origin, direction = result
            
            # Clip to polygon 1
            seg1 = clip_line_to_polygon_2d(origin, direction, p1['hull2d'], p1['u'], p1['v'])
            if seg1 is None:
                seg1 = clip_line_to_polygon_2d(origin, -direction, p1['hull2d'], p1['u'], p1['v'])
                if seg1 is not None:
                    seg1 = (seg1[1], seg1[0])
            
            # Clip to polygon 2
            seg2 = clip_line_to_polygon_2d(origin, direction, p2['hull2d'], p2['u'], p2['v'])
            if seg2 is None:
                seg2 = clip_line_to_polygon_2d(origin, -direction, p2['hull2d'], p2['u'], p2['v'])
                if seg2 is not None:
                    seg2 = (seg2[1], seg2[0])
            
            if seg1 is None or seg2 is None:
                continue
            
            # Find overlap between the two segments
            # Projects segments to parameter t along direction
            t_start = max(
                np.dot(seg1[0][:2] - origin[:2], direction[:2]) / max(np.linalg.norm(direction[:2]), 1e-10),
                np.dot(seg2[0][:2] - origin[:2], direction[:2]) / max(np.linalg.norm(direction[:2]), 1e-10)
            )
            t_end = min(
                np.dot(seg1[1][:2] - origin[:2], direction[:2]) / max(np.linalg.norm(direction[:2]), 1e-10),
                np.dot(seg2[1][:2] - origin[:2], direction[:2]) / max(np.linalg.norm(direction[:2]), 1e-10)
            )
            
            if t_end <= t_start:
                continue
            
            edge_start = origin + direction * t_start
            edge_end = origin + direction * t_end
            edge_len = float(np.linalg.norm(edge_end - edge_start))
            
            if edge_len < 0.3:  # Skip tiny intersections
                continue
            
            # Classify edge type
            n_mid = (p1['n'] + p2['n']) / 2
            is_convex = n_mid[2] > 0  # Both normals point somewhat up
            
            # Determine if horizontal (ridge) or sloped (hip/valley)
            dz = abs(edge_end[2] - edge_start[2])
            horizontalness = dz / max(edge_len, 0.01)
            
            if horizontalness < 0.1:  # <10% slope = nearly horizontal
                etype = 'h' if is_convex else 'u'  # horizontal concave = valley
            else:
                etype = 'n' if is_convex else 'u'  # sloped convex = hip, concave = valley
            
            intersections.append({
                'planes': [i, j],
                'type': etype,
                'length_m': round(edge_len, 3),
                'start': [round(float(edge_start[0]), 3), round(float(edge_start[1]), 3), round(float(edge_start[2]), 3)],
                'end': [round(float(edge_end[0]), 3), round(float(edge_end[1]), 3), round(float(edge_end[2]), 3)],
            })
    
    return intersections
