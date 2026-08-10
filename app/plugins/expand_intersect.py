"""Polygon expansion + plane intersection clipping.
Expands each plane uphill + sideways, clips by neighbor intersection lines.
"""
import numpy as np
from scipy.spatial import ConvexHull

def _clip_2d(poly_2d, line_pt, line_dir, keep_left=True):
    """Clip 2D polygon by a line. keep_left: True = keep points with positive cross."""
    if len(poly_2d) < 3:
        return poly_2d
    n = np.array([-line_dir[1], line_dir[0]])
    n /= np.linalg.norm(n)
    if not keep_left:
        n = -n
    
    result = []
    nv = len(poly_2d)
    for i in range(nv):
        cur = np.array(poly_2d[i])
        nxt = np.array(poly_2d[(i+1) % nv])
        dc = np.dot(cur - line_pt, n)
        dn = np.dot(nxt - line_pt, n)
        if dc >= -0.01:
            result.append(cur.tolist())
        if dc * dn < 0:
            t = dc / (dc - dn) if abs(dc - dn) > 1e-10 else 0
            result.append((cur + t * (nxt - cur)).tolist())
    return np.array(result) if len(result) >= 3 else poly_2d

def expand_and_intersect(planes):
    """Expand polygons uphill + sideways, clip by neighbor intersections."""
    if len(planes) < 2:
        return planes
    
    # Step 1: Build plane equations and 2D bases
    for p in planes:
        verts = np.array(p['vertices_3d'])
        n_ = p.get('_normal')
        if n_ is None:
            n_ = np.array([0, 0, 1])
        centroid = verts.mean(axis=0)
        d_ = np.dot(n_, centroid)
        
        # Build 2D basis: u = eave dir (horizontal), v = uphill (in-plane, perpendicular to u)
        z_axis = np.array([0., 0., 1.])
        u = np.cross(n_, z_axis)
        un = np.linalg.norm(u)
        u = u / un if un > 1e-10 else np.array([1., 0., 0.])
        v = np.cross(n_, u)
        v /= np.linalg.norm(v)
        
        p['_n'] = n_
        p['_d'] = d_
        p['_u'] = u
        p['_v'] = v
        p['_c'] = centroid
    
    # Step 2: Find plane adjacency (different normals, close centroids)
    adjacency = []
    for i in range(len(planes)):
        for j in range(i+1, len(planes)):
            n1, n2 = planes[i]['_n'], planes[j]['_n']
            dot = abs(np.dot(n1, n2))
            if 0.1 < dot < 0.95:
                dist = np.linalg.norm(planes[i]['_c'] - planes[j]['_c'])
                if dist < 15:
                    adjacency.append((i, j))
    
    # Step 3: For each plane pair, compute 3D intersection line
    intersections = {}  # (pi, pj) -> (point_3d, dir_3d)
    for pi, pj in adjacency:
        n1, c1 = planes[pi]['_n'], planes[pi]['_c']
        n2, c2 = planes[pj]['_n'], planes[pj]['_c']
        d1 = np.dot(n1, c1)
        d2 = np.dot(n2, c2)
        direction = np.cross(n1, n2)
        dnorm = np.linalg.norm(direction)
        if dnorm < 1e-6:
            continue
        direction /= dnorm
        # Find point on line
        A = np.array([n1, n2, direction])
        b = np.array([d1, d2, np.dot(direction, (c1+c2)/2)])
        try:
            point = np.linalg.solve(A, b)
        except:
            continue
        intersections[(pi, pj)] = (point, direction)
    
    # Step 4: Expand each polygon uphill (+v) and sideways (+-u)
    for pi, p in enumerate(planes):
        verts = np.array(p['vertices_3d'])
        u, v = p['_u'], p['_v']
        c = p['_c']
        
        # Project to 2D
        pts_2d = np.column_stack([np.dot(verts[:, :2], u[:2]), np.dot(verts[:, :2], v[:2])])
        cf_2d = np.array([np.dot(c[:2], u[:2]), np.dot(c[:2], v[:2])])
        
        # Convex hull
        try:
            hull = ConvexHull(pts_2d)
            poly_2d = pts_2d[hull.vertices]
        except:
            poly_2d = pts_2d
        
        # Expand uphill (increase v) and sideways (expand u range)
        vmin = poly_2d[:, 1].min()
        vmax = poly_2d[:, 1].max()
        umin = poly_2d[:, 0].min()
        umax = poly_2d[:, 0].max()
        urange = umax - umin
        
        # Expand: extend v upward by 30% of height, u sideways by 20%
        expand_v = (vmax - vmin) * 0.5
        expand_u = urange * 0.3
        
        # Build expanded bbox corners
        expanded = np.array([
            [umin - expand_u, vmin],
            [umin - expand_u, vmax + expand_v],
            [umax + expand_u, vmax + expand_v],
            [umax + expand_u, vmin],
        ])
        
        # Clip expanded polygon by intersection lines from neighbors
        for (a, b), (line_pt, line_dir) in intersections.items():
            if a != pi and b != pi:
                continue
            # Project intersection to this plane's 2D
            lp_2d = np.array([np.dot(line_pt[:2], u[:2]), np.dot(line_pt[:2], v[:2])])
            ld_2d = np.array([np.dot(line_dir[:2], u[:2]), np.dot(line_dir[:2], v[:2])])
            ldn = np.linalg.norm(ld_2d)
            if ldn < 1e-6:
                continue
            ld_2d /= ldn
            
            # Determine which side to keep (toward centroid)
            side = np.dot(cf_2d - lp_2d, np.array([-ld_2d[1], ld_2d[0]]))
            expanded = _clip_2d(expanded, lp_2d, ld_2d, keep_left=side > 0)
        
        # Also clip to keep eave (bottom edge)
        # The eave is the lowest v edge
        eave_v = vmin
        expanded = _clip_2d(expanded, np.array([umin, eave_v]), np.array([1., 0.]), keep_left=(cf_2d[1] > eave_v))
        
        # Simplify to convex hull
        try:
            hull2 = ConvexHull(expanded)
            poly_2d = expanded[hull2.vertices]
        except:
            poly_2d = expanded
        
        # Simplify
        import cv2
        contour = poly_2d[:, :2].astype(np.float32).reshape(-1, 1, 2)
        diag = np.linalg.norm(poly_2d.max(axis=0) - poly_2d.min(axis=0))
        simp = cv2.approxPolyDP(contour, diag * 0.05, True).reshape(-1, 2)
        if len(simp) >= 3:
            poly_2d = simp
        
        # Convert back to 3D
        n_, d_ = p['_n'], p['_d']
        verts_out = []
        for pt2 in poly_2d:
            xy = pt2[0] * u[:2] + pt2[1] * v[:2]
            if abs(n_[2]) > 1e-6:
                z = (d_ - n_[0]*xy[0] - n_[1]*xy[1]) / n_[2]
            else:
                z = p['_c'][2]
            verts_out.append([round(float(xy[0]),3), round(float(xy[1]),3), round(float(z),3)])
        
        p['vertices_3d'] = verts_out
    
    # Clean up
    for p in planes:
        for key in ['_n', '_d', '_u', '_v', '_c']:
            p.pop(key, None)
    
    return planes
