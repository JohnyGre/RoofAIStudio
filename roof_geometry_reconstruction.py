# -*- coding: utf-8 -*-
"""
roof_geometry_reconstruction.py — rekonštrukcia geometrie strešných rovín z LiDAR zhlukov.

Namiesto fan-triangulácie mračna:
  1. plane fitting (SVD) pre každý zhluk
  2. susedstvo rovín (KD-tree, eps)
  3. priesečnica rovín + klasifikácia hrany (hrebeň/nárožie/úžľabie)
  4. rekonštrukcia čistého polygónu (priesečníkové polroviny na hull bunkách)
  5. export: vrcholy + hrany s metadátami (typ, dĺžka, p1, p2)
  6. fallback: pôvodný postup pre nezvyčajné tvary (appka nespadne)
"""
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon as ShPolygon, Point, LineString, MultiPoint
from shapely import concave_hull
from shapely.ops import unary_union

EPS_ADJ = 0.3          # m — max vzdialenosť susedných bodov rovín
MIN_PAIRS = 20         # min. počet blízkych párov pre susedstvo
SHARE_FRAC = 0.05      # min. podiel bodov roviny do 1.5 m od priesečnice
HOLE_H = 0.15          # priesečnica je "vodorovná", ak |s.z| < HOLE_H
SIMPLE_HULL = 0.35     # m — zjednodušenie hull buniek (rovné hrany)
MIN_SEG = 0.5          # m — min. dĺžka úseku priesečnice


def fit_plane(points):
    """LSQ/SVD rovina cez body. Vráti dict s normal, d, centroid, pts."""
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]
    if normal[2] < 0:  # normála nahor
        normal = -normal
    d = float(normal @ centroid)
    if abs(normal[2]) < 1e-6:
        normal = normal.copy(); normal[2] = 1e-6
    coef = np.array([-normal[0] / normal[2], -normal[1] / normal[2], d / normal[2]])
    return {'normal': normal, 'd': d, 'centroid': centroid, 'pts': pts, 'coef': coef}


def detect_adjacency(planes, eps=EPS_ADJ, min_pairs=MIN_PAIRS):
    """Susedstvo: body z i s najbližším susedom v j do eps (2D). Vráti set párov (i, j)."""
    n = len(planes)
    trees = [cKDTree(pl['pts'][:, :2]) for pl in planes]
    adj = set()
    for i in range(n):
        for j in range(i + 1, n):
            d, _ = trees[j].query(planes[i]['pts'][:, :2], k=1)
            if (d < eps).sum() >= min_pairs:
                adj.add((i, j))
    return adj


def edge_line(pi, pj):
    """Priesečnica dvoch rovín (3D): bod p + jednotkový smer s. None ak rovnobežné."""
    n1, n2 = pi['normal'], pj['normal']
    s = np.cross(n1, n2)
    sl = np.linalg.norm(s)
    if sl < 1e-9:
        return None
    s = s / sl
    c = (pi['centroid'] + pj['centroid']) / 2
    A = np.vstack([n1, n2, s])
    b = np.array([pi['d'], pj['d'], s @ c])
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p, s


def classify_edge(p, s, pi, pj):
    """Klasifikácia priesečnice: hrebeň / nárožie / úžľabie."""
    horizontal = abs(s[2]) < HOLE_H
    s2 = s[:2] / max(np.linalg.norm(s[:2]), 1e-12)
    ci = pi['centroid'][:2] - p[:2]
    cj = pj['centroid'][:2] - p[:2]
    si = float(ci[0] * s2[1] - ci[1] * s2[0])
    sj = float(cj[0] * s2[1] - cj[1] * s2[0])
    opposite = (si * sj) < 0  # roviny na opačných stranách priamky = konvexná hrana
    if opposite:
        return 'hrebeň' if horizontal else 'nárožie'
    return 'úžľabie'


def plane_edges(planes, adj, outline):
    """Pre každý susedný pár: klasifikácia + 2D úsek priesečnice v rámci obrysu.
    Vráti edges: dict (i, j) -> {'seg': LineString, 'type': str, 'p': 3D, 's': 3D}."""
    edges = {}
    for (i, j) in adj:
        e = edge_line(planes[i], planes[j])
        if e is None:
            continue
        p, s = e
        try:
            typ = classify_edge(p, s, planes[i], planes[j])
        except Exception:
            continue
        seg = segment_in_outline(p, s, outline)
        if seg is None:
            continue
        edges[(i, j)] = {'seg': seg, 'type': typ, 'p': p, 's': s}
    return edges


def segment_in_outline(p, s, outline, t_range=60.0):
    """Úsek priesečnice v rámci obrysu (2D). None ak príliš krátky/mimo."""
    s2 = s[:2]
    sl = np.linalg.norm(s2)
    if sl < 1e-9:
        return None
    s2 = s2 / sl
    p2 = np.asarray(p[:2], dtype=float)
    line = LineString([p2 - t_range * s2, p2 + t_range * s2])
    seg = line.intersection(outline)
    if seg.is_empty or seg.length < MIN_SEG:
        return None
    return seg


def reconstruct_plane(i, planes, edges, outline, min_area=0.5, poly_tol=0.35):
    """Čistý polygón roviny i: hull bodov (priamky po simplify) orezaný
    priesečníkovými polrovinami so susedmi. Vráti (ShPolygon, list hrán) alebo None.
    Hrany: {'type', 'length_m', 'p1', 'p2'} (3D)."""
    pl = planes[i]
    try:
        hull = concave_hull(MultiPoint(pl['pts'][:, :2]), ratio=0.05).simplify(SIMPLE_HULL, preserve_topology=True)
    except Exception:
        hull = ShPolygon(pl['pts'][:, :2]).convex_hull
    if hull.geom_type == 'MultiPolygon':
        hull = max(hull.geoms, key=lambda g: g.area)
    if hull.geom_type != 'Polygon':
        return None
    poly = hull

    # priesečníkové polroviny od susedov
    for (a, b), e in edges.items():
        if i not in (a, b):
            continue
        seg = e['seg']
        coords = list(seg.coords)
        (x0, y0), (x1, y1) = coords[0], coords[-1]
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L < 1e-9:
            continue
        nx, ny = -dy / L, dx / L
        ctr = np.array([pl['centroid'][0], pl['centroid'][1]])
        if np.dot([nx, ny], ctr - np.array([x0, y0])) < 0:
            nx, ny = -nx, -ny
        half = ShPolygon([
            (x0 - 50 * nx, y0 - 50 * ny), (x1 - 50 * nx, y1 - 50 * ny),
            (x1 + 50 * nx, y1 + 50 * ny), (x0 + 50 * nx, y0 + 50 * ny)])
        poly = poly.intersection(half)
        if poly.is_empty:
            return None
    try:
        poly = poly.simplify(poly_tol, preserve_topology=True)
    except Exception:
        return None
    if poly.geom_type != 'Polygon' or not poly.is_valid or poly.area < min_area:
        return None

    # hrany polygónu + klasifikácia (3D)
    coef = pl['coef']
    coords = list(poly.exterior.coords)[:-1]
    z_of = lambda p: float(coef[0] * p[0] + coef[1] * p[1] + coef[2])
    zs = [z_of(p) for p in coords]
    zmin, zmax = min(zs), max(zs)
    out_edges = []
    kk = len(coords)
    for t in range(kk):
        p1 = coords[t]; p2 = coords[(t + 1) % kk]
        z1 = z_of(p1); z2 = z_of(p2)
        m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (z1 + z2) / 2)
        length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        dz = abs(z2 - z1)
        # typ: priesečník so susedom alebo okrajová hrana (odkvap/štít)
        typ = None
        best_d = 1.2
        for (a, b), e in edges.items():
            if i not in (a, b):
                continue
            d = _dist_point_line_2d(m[:2], e['p'][:2], e['s'][:2])
            if d < best_d:
                best_d = d
                typ = e['type']
        if typ is None:
            # okrajová hrana: odkvap (min Z) / štít (inak)
            zrel = (m[2] - zmin) / (zmax - zmin + 1e-9)
            typ = 'odkvap' if zrel < 0.3 else 'štít'
        out_edges.append({'type': typ, 'length_m': round(length, 2),
                          'p1': [round(float(p1[0]), 2), round(float(p1[1]), 2), round(z1, 2)],
                          'p2': [round(float(p2[0]), 2), round(float(p2[1]), 2), round(z2, 2)]})
    # 3D vrcholy
    verts3d = [[float(p[0]), float(p[1]), round(z_of(p), 2)] for p in coords]
    return ShPolygon(coords), out_edges, verts3d


def _dist_point_line_2d(pt, p0, s2):
    """Vzdialenosť bodu od priamky (2D), s2 jednotkový smer."""
    v = np.array(pt) - np.asarray(p0[:2])
    s2 = np.asarray(s2[:2])
    sl = np.linalg.norm(s2)
    if sl < 1e-12:
        return float(np.linalg.norm(v))
    s2 = s2 / sl
    return abs(float(v[0] * s2[1] - v[1] * s2[0]))


def reconstruct_all(planes_raw, outline, min_area=0.5):
    """Hlavné API: planes_raw = list numpy (N,3) zhlukov.
    Vráti: (planes, results, warnings)
      planes  — zoznam dictov s normal/d/centroid/pts
      results — list dictov {poly (ShPolygon), edges (hrany), verts3d, plane_idx}
                len pre úspešné; neúspešné sú v warnings
      warnings — list str
    outline: ShPolygon (naklikaný/OSM obrys, S-JTSK)."""
    warnings = []
    planes = [fit_plane(np.asarray(p, dtype=float)) for p in planes_raw]
    adj = detect_adjacency(planes)
    edges = plane_edges(planes, adj, outline)
    results, warnings = [], []
    for i, pl in enumerate(planes):
        try:
            rec = reconstruct_plane(i, planes, edges, outline, min_area=min_area)
        except Exception as ex:
            rec = None
            warnings.append(f'R{i+1}: rekonštrukcia zlyhala ({ex}) — fallback na pôvodný postup')
        if rec is None:
            warnings.append(f'R{i+1}: nepodaril sa konvexný polygón — fallback')
            continue
        poly, edges_meta, verts3d = rec
        results.append({'poly': poly, 'edges': edges_meta, 'verts3d': verts3d, 'plane_idx': i})
    return planes, results, warnings
