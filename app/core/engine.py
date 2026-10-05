# -*- coding: utf-8 -*-
"""v3 geometrický engine — JEDINÉ miesto, kde vzniká geometria.

Deterministický (pevný seed), testovateľný, bez GUI a bez LLM.
Reťaz: mračno → segmentácia rovín (RANSAC) → polygóny (concave hull) →
klasifikácia hrán (o/h/n/u/s) podľa pravidiel projektu:
  * odkvap (o)   = najnižšia hrana roviny,
  * hrebeň (h)   = najvyššia hrana ležiaca na priesečnici so susedom,
  * nárožie (n)  = šikmá hrana na priesečnici, uhol od napojeného odkvapu < 90°,
  * úžľabie (u)  = to isté, ale > 90° (konkávne),
  * štít (s)     = zvyšná hrana (nie odkvap, nie priesečnica).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULTS = dict(
    distance_threshold=0.06,   # m
    min_inliers=150,
    max_planes=12,
    min_normal_z=0.15,         # len strechovité roviny
    seed=42,
    polygon_tolerance=1.5,     # m, simplifikácia
    edge_on_intersection_tol=0.30,
    min_shared_edge_m=1.0,
    min_plane_points=150,
    min_edge_m=2.5,
)



def separate_buildings(points: np.ndarray, target_xy, cell: float = 0.8, min_points: int = 150) -> np.ndarray:
    """Rozdelí body na jednotlivé budovy (2D connected components) a vráti tú pri cieli.

    Prečo: v rádiuse 40 m je aj susedná budova; bez oddelenia engine pomieša roviny
    dvoch objektov (dôkaz: súčet plôch 3897 m² vs pôdorys 343,69 m²).
    """
    from scipy import ndimage

    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    mins = pts[:, :2].min(axis=0) - cell
    idx = np.floor((pts[:, :2] - mins) / cell).astype(np.int64)
    shape = tuple((idx.max(axis=0) + 3).tolist())
    grid = np.zeros(shape, dtype=bool)
    grid[idx[:, 0], idx[:, 1]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
    if n <= 1:
        return pts
    lbl_pts = lab[idx[:, 0], idx[:, 1]]
    best_id, best_d = None, None
    for k in range(1, n + 1):
        m = lbl_pts == k
        if int(m.sum()) < min_points:
            continue
        sub = pts[m][:, :2]
        d = float(np.sqrt(((sub - np.asarray(target_xy, dtype=float)) ** 2).sum(axis=1)).min())
        if best_id is None or d < best_d:
            best_id, best_d = k, d
        if d <= cell:      # budova obsahujúca cieľový bod má prednosť
            best_id, best_d = k, d
            break
    if best_id is None:
        return pts
    return pts[lbl_pts == best_id]


def _merge_edges(edges, angle_tol_deg: float = 20.0):
    """Spojí susedné hrany rovnakého typu, ak sú takmer rovnobežné (menej fragmentov)."""
    out = []
    for e in edges:
        if out and out[-1]["type"] == e["type"]:
            a = out[-1]
            d1 = np.array([a["end"][0] - a["start"][0], a["end"][1] - a["start"][1]])
            d2 = np.array([e["end"][0] - e["start"][0], e["end"][1] - e["start"][1]])
            l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if l1 > 1e-9 and l2 > 1e-9:
                ang = math.degrees(math.acos(min(1.0, abs(float((d1 / l1).dot(d2 / l2))))))
                if ang < angle_tol_deg:
                    a["end"] = e["end"]
                    a["length_m"] = round(a["length_m"] + e["length_m"], 3)
                    continue
        out.append(dict(e))
    return out


# ─── segmentácia ─────────────────────────────────────────────────────────────

def _fit_plane_svd(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c)
    n = vh[-1]
    if n[2] < 0:
        n = -n
    return n, -float(n.dot(c))


def _ransac_plane(pts: np.ndarray, thr: float, iters: int, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    n = len(pts)
    if n < 3:
        return None
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nv = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nv)
        if ln < 1e-9:
            continue
        nv = nv / ln
        d = -float(nv.dot(p0))
        dist = np.abs(pts @ nv + d)
        inl = np.where(dist < thr)[0]
        if best is None or len(inl) > len(best[2]):
            best = (nv, d, inl)
    if best is None or len(best[2]) < 3:
        return None
    nv, d = _fit_plane_svd(pts[best[2]])
    dist = np.abs(pts @ nv + d)
    inl = np.where(dist < thr)[0]
    return nv, d, inl


def segment_planes(points: np.ndarray, **kw) -> List[Dict[str, Any]]:
    """Rozdelí mračno na roviny. Vracia zoznam dictov (normal, d, inliers, ...)."""
    cfg = {**DEFAULTS, **kw}
    rng = np.random.default_rng(cfg["seed"])
    pts = np.asarray(points, dtype=float)
    remaining = np.arange(len(pts))
    planes: List[Dict[str, Any]] = []

    while len(remaining) > cfg["min_inliers"] and len(planes) < cfg["max_planes"]:
        res = _ransac_plane(pts[remaining], cfg["distance_threshold"], 600, rng)
        if res is None:
            break
        nv, d, inl = res
        if abs(nv[2]) < cfg["min_normal_z"]:
            break
        idx = remaining[inl]
        if len(idx) < cfg["min_inliers"]:
            break
        pl = pts[idx]
        nv, d = _fit_plane_svd(pl)
        dist = np.abs(pl @ nv + d)
        rmse = float(np.sqrt((dist ** 2).mean()))
        slope = math.degrees(math.acos(min(1.0, abs(float(nv[2])))))
        az = (math.degrees(math.atan2(nv[0], nv[1])) + 360.0) % 360.0
        planes.append({
            "normal": nv, "d": float(d), "points": pl,
            "area_m2": float(len(pl)),
            "rmse_m": round(rmse, 4),
            "slope_deg": round(slope, 2),
            "azimuth_deg": round(az, 1),
            "n_points": int(len(pl)),
        })
        remaining = remaining[~np.isin(remaining, idx)]
    return planes



def split_plane_components(planes, cell: float = 0.5, min_points: int = 150):
    """Rozdelí rovinu na priestorovo spojité časti.

    Poučenie z projektu: RANSAC zlepí dve fyzicky oddelené plochy s rovnakým
    sklonom (napr. ploché strechy viacerých domov v rade). Riešenie: connected
    components v 2D mriežke.
    """
    from scipy import ndimage

    out = []
    for pl in planes:
        pts = pl["points"]
        if len(pts) < 2 * min_points:
            out.append(pl)
            continue
        mins = pts[:, :2].min(axis=0) - cell
        idx = np.floor((pts[:, :2] - mins) / cell).astype(np.int64)
        shape = tuple((idx.max(axis=0) + 3).tolist())
        grid = np.zeros(shape, dtype=bool)
        grid[idx[:, 0], idx[:, 1]] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
        if n <= 1:
            out.append(pl)
            continue
        lbl_pts = lab[idx[:, 0], idx[:, 1]]
        pieces = 0
        for k in range(1, n + 1):
            m = lbl_pts == k
            sub = pts[m]
            if len(sub) < min_points:
                continue
            nv, d = _fit_plane_svd(sub)
            dist = np.abs(sub @ nv + d)
            out.append({
                "normal": nv, "d": float(d), "points": sub, "n_points": int(len(sub)),
                "rmse_m": round(float(np.sqrt((dist ** 2).mean())), 4),
                "slope_deg": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
                "azimuth_deg": round((math.degrees(math.atan2(nv[0], nv[1])) + 360.0) % 360.0, 1),
                "area_m2": float(len(sub)),
            })
            pieces += 1
        if pieces == 0:
            out.append(pl)
    return out


def merge_coplanar_planes(planes, angle_tol_deg: float = 7.0, dist_tol_m: float = 0.15):
    """Spojí roviny, ktoré sú v podstate tá istá strešná plocha.

    Prečo: RANSAC rozdelí jednu plochu na viac kusov (napr. az 318,5° a 318,4°),
    čo vyrába zbytočné hrany. Iteruje, kým sa niečo spája.
    """
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(planes)
        for i in range(len(planes)):
            if used[i]:
                continue
            ni = planes[i]["normal"]
            acc = planes[i]
            for j in range(i + 1, len(planes)):
                if used[j]:
                    continue
                nj = planes[j]["normal"]
                ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(ni, nj))))))
                if ang > angle_tol_deg:
                    continue
                # sú body druhej roviny blízko roviny prvej?
                pts_j = planes[j]["points"]
                dist = np.abs(pts_j @ acc["normal"] + acc["d"])
                if float(np.percentile(dist, 90)) > dist_tol_m:
                    continue
                # a sú priestorovo pri sebe? (inak by sa zliali ploché strechy celého radu)
                if not _spatially_adjacent(acc["points"], pts_j, max_dist_m=0.8):
                    continue
                acc = _merge_two_planes(acc, planes[j])
                used[j] = True
                ni = acc["normal"]
                changed = True
            used[i] = True
            out.append(acc)
        planes = out
    return planes


def _spatially_adjacent(pts_a, pts_b, max_dist_m: float = 0.8) -> bool:
    """Susedia dve množiny bodov v pôdoryse (XY)?"""
    from scipy.spatial import cKDTree

    if len(pts_a) == 0 or len(pts_b) == 0:
        return False
    tree = cKDTree(pts_a[:, :2])
    d, _ = tree.query(pts_b[:, :2], k=1)
    return bool(np.min(d) <= max_dist_m)


def _merge_two_planes(a, b):
    pts = np.vstack([a["points"], b["points"]])
    n, d = _fit_plane_svd(pts)
    dist = np.abs(pts @ n + d)
    rmse = float(np.sqrt((dist ** 2).mean()))
    slope = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
    az = (math.degrees(math.atan2(n[0], n[1])) + 360.0) % 360.0
    return {"normal": n, "d": float(d), "points": pts, "area_m2": float(len(pts)),
            "rmse_m": round(rmse, 4), "slope_deg": round(slope, 2), "azimuth_deg": round(az, 1),
            "n_points": int(len(pts))}


def prune_small_planes(planes, min_points: int = 400):
    """Vyhodí drobné plochy (komíny, vikiere, šum) — znižuje počet rovín aj hrán."""
    return [p for p in planes if p["n_points"] >= min_points]


def _turn_angle_deg(prev, cur, nxt) -> float:
    """Uhol zlomu v bode `cur` (0° = ide rovno, 90° = pravý uhol)."""
    v1 = np.array([cur[0] - prev[0], cur[1] - prev[1]])
    v2 = np.array([nxt[0] - cur[0], nxt[1] - cur[1]])
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if l1 < 1e-9 or l2 < 1e-9:
        return 0.0
    cosang = float(np.clip((v1 / l1).dot(v2 / l2), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _shorten_ring(verts2, min_edge_m: float = 2.0, straight_tol_deg: float = 30.0):
    """Vypustí vrcholy, ktoré netvoria roh a tvoria krátke úseky → menej čiar.

    Rovné úseky (zlom < straight_tol_deg) sa spoja do jednej hrany, presne ako
    to má byť pri streche: hrany sú línie medzi rohmi, nie mnoho krátkych kúskov.
    """
    if len(verts2) < 4:
        return verts2
    out = list(verts2)
    changed = True
    while changed and len(out) > 3:
        changed = False
        n = len(out)
        for i in range(n):
            prev, cur, nxt = out[(i - 1) % n], out[i], out[(i + 1) % n]
            short = math.hypot(cur[0] - prev[0], cur[1] - prev[1]) < min_edge_m
            straight = _turn_angle_deg(prev, cur, nxt) < straight_tol_deg
            if short or straight:
                out.pop(i)
                changed = True
                break
    return out if len(out) >= 3 else verts2

# ─── polygóny a orezávanie ────────────────────────────────────────────────────

def plane_polygon(plane: Dict[str, Any], tolerance: float = 1.5) -> Optional[List[List[float]]]:
    """Polygón roviny v 2D (X, Y) — concave hull, orezaný na pravidelný tvar."""
    from shapely import concave_hull
    from shapely.geometry import MultiPoint

    pts = plane["points"]
    if len(pts) < 4:
        return None
    mp = MultiPoint([(float(p[0]), float(p[1])) for p in pts])
    poly = None
    for ratio in (0.25, 0.4, 0.6):
        try:
            cand = concave_hull(mp, ratio=ratio)
            if cand.geom_type == "Polygon" and cand.area > 0:
                poly = cand
                break
        except Exception:
            continue
    if poly is None:
        poly = mp.convex_hull
    if poly.geom_type != "Polygon":
        poly = poly.convex_hull

    try:
        rect = poly.minimum_rotated_rectangle
        if rect.geom_type == "Polygon" and 0 < rect.area <= poly.area * 1.08:
            poly = rect
        else:
            ch = poly.convex_hull
            if ch.area <= poly.area * 1.12:
                poly = ch
    except Exception:
        pass

    poly = poly.simplify(tolerance, preserve_topology=True)
    pts2 = list(poly.exterior.coords)[:-1]
    if len(pts2) < 3:
        return None
    return [[float(x), float(y)] for x, y in pts2]


def plane_z_at(plane: Dict[str, Any], x: float, y: float) -> float:
    n = plane["normal"]
    if abs(n[2]) < 1e-9:
        return float(plane["points"][:, 2].mean())
    return float(-(n[0] * x + n[1] * y + plane["d"]) / n[2])


def _clip_polygon_by_line(poly, p0, direction, keep_sign: int):
    """Sutherland–Hodgman: orež polygón priamkou (p0, direction) v 2D."""
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        sc = keep_sign * (direction[0] * (cur[1] - p0[1]) - direction[1] * (cur[0] - p0[0]))
        sn = keep_sign * (direction[0] * (nxt[1] - p0[1]) - direction[1] * (nxt[0] - p0[0]))
        if sc >= 0:
            out.append(cur)
        if (sc >= 0) != (sn >= 0):
            t = sc / (sc - sn) if (sc - sn) != 0 else 0.0
            out.append([cur[0] + t * (nxt[0] - cur[0]), cur[1] + t * (nxt[1] - cur[1])])
    return out if len(out) >= 3 else None


def _adjacent_pairs(planes, near_tol: float = 0.50, min_pts: int = 3):
    """Overené susedstvo: priesečnica + dosť bodov oboch rovín blízko nej."""
    from shapely.geometry import LineString, Point

    pairs = {}
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            line = _intersection_line(planes[i]["normal"], planes[i]["d"],
                                      planes[j]["normal"], planes[j]["d"])
            if not line:
                continue
            point, direction = line
            if abs(direction[2]) > 0.95:            # zvislá priesečnica — nezaujíma
                continue
            # takmer rovnobežné roviny majú nestabilnú priesečnicu → preskočiť
            ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(planes[i]["normal"], planes[j]["normal"]))))))
            if ang < 3.0:
                continue
            d_xy = np.array([direction[0], direction[1]])
            ln = np.linalg.norm(d_xy)
            if ln < 1e-9:
                continue
            d_xy = d_xy / ln
            p0 = np.array([point[0], point[1]])

            def near_count(pl):
                p = pl["points"][:, :2]
                cross = np.abs(d_xy[0] * (p[:, 1] - p0[1]) - d_xy[1] * (p[:, 0] - p0[0]))
                return int((cross < near_tol).sum())

            if near_count(planes[i]) >= min_pts and near_count(planes[j]) >= min_pts:
                pairs[(i, j)] = {"point": p0, "dir": d_xy, "point3": point, "dir3": direction}
    return pairs


def resolve_overlaps(planes, pairs) -> None:
    """Oreže polygón každej roviny polrovinami susedov (bez presahov a medzier)."""
    for (i, j), info in pairs.items():
        for own, other in ((i, j), (j, i)):
            poly = planes[own].get("vertices_2d")
            if not poly or len(poly) < 3:
                continue
            pts = planes[own]["points"][:, :2]
            mean = pts.mean(axis=0)
            d = info["dir"]
            side = d[0] * (mean[1] - info["point"][1]) - d[1] * (mean[0] - info["point"][0])
            keep = 1 if side >= 0 else -1
            clipped = _clip_polygon_by_line(poly, info["point"], d, keep)
            if clipped and len(clipped) >= 3:
                planes[own]["vertices_2d"] = clipped


# ─── klasifikácia hrán ───────────────────────────────────────────────────────

def _intersection_line(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float):
    direction = np.cross(n1, n2)
    ln = np.linalg.norm(direction)
    if ln < 1e-8:
        return None
    direction = direction / ln
    A = np.vstack([n1, n2, direction])
    b = np.array([-d1, -d2, 0.0])
    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return point, direction


def _dist_to_line(p, point, direction) -> float:
    v = np.asarray(p, dtype=float) - np.asarray(point, dtype=float)
    return float(np.linalg.norm(np.cross(v, direction)))


def _down_dir(normal) -> np.ndarray:
    """Vodorovný smer spádu: z = -(n·xy + d)/n_z → spád ide v smere (n_x, n_y)."""
    v = np.array([normal[0], normal[1]], dtype=float)
    l = np.linalg.norm(v)
    return v / l if l > 1e-9 else v


def _plane_trees(planes):
    """KD-stromy (XY) pre každú rovinu — na zistenie, ktorá plocha v bode naozaj leží."""
    from scipy.spatial import cKDTree

    return [cKDTree(pl["points"][:, :2]) for pl in planes]


def _fold_convex(plane_i, plane_j, point, direction, sample_m: float = 0.6, trees=None, idx=None):
    """Konvexnosť z REÁLNYCH bodov mračna (nie z extrapolácie rovín).

    Vezmeme body oboch plôch vo vzdialenosti 0,3–1,5 m od priesečnice (v pôdoryse),
    pre každý spočítame Δz = z_bodu − z_línie v mieste jeho priemetu a z mediánu
    rozhodneme: Δz < 0 → hrebeň/nárožie (konvexné), Δz > 0 → úžľabie (konkávne).
    """
    d3 = np.asarray(direction, dtype=float)
    d3 = d3 / (np.linalg.norm(d3) or 1.0)
    p3 = np.asarray(point, dtype=float)
    d2 = d3[:2]
    ln = np.linalg.norm(d2)
    if ln < 1e-9:
        return None
    d2 = d2 / ln

    deltas = []
    for pl in (plane_i, plane_j):
        P = pl["points"]
        if len(P) < 5:
            continue
        v = P[:, :2] - p3[:2]
        perp = np.abs(d2[0] * v[:, 1] - d2[1] * v[:, 0])      # kolmá vzdialenosť v pôdoryse
        sel = (perp > 0.3) & (perp < 1.5)
        if int(sel.sum()) < 5:
            continue
        sub = P[sel]
        t = (sub[:, :2] - p3[:2]) @ d2                          # poloha pozdĺž línie
        z_line = p3[2] + t * d3[2]                              # výška línie v tom mieste
        deltas.extend((sub[:, 2] - z_line).tolist())

    if len(deltas) < 8:
        return None
    med = float(np.median(deltas))
    if med < -0.05:
        return True
    if med > 0.05:
        return False
    return None


def _ridge_or_hip(pi, pj) -> str:
    di, dj = _down_dir(pi["normal"]), _down_dir(pj["normal"])
    a = math.degrees(math.acos(min(1.0, max(-1.0, float(di.dot(dj))))))
    return "h" if a > 135.0 else "n"


def classify_plane_subtype(pl) -> str:
    verts = pl.get("vertices_2d") or []
    edges = pl.get("edges") or []
    types = [e["type"] for e in edges]
    slope = float(pl.get("slope_deg") or 0.0)
    area = float(pl.get("area_m2") or 0.0)
    if slope < 8.0:
        return "plochá"
    if area < 15.0 and slope >= 20.0:
        return "vikier"
    if len(verts) <= 3 and types.count("n") >= 2:
        return "valba (trojuholníková)"
    return "sedlová/valbová"


def classify_edges(planes, **kw) -> None:
    """Typy hrán: o/h/n/u/s. Hrany ležiace na overenej priesečnici so susedom
    dostanú typ podľa konvexnosti (h/n = konvexné, u = konkávne), najnižšia
    hrana roviny je odkvap (o), zvyšok štít (s)."""
    cfg = {**DEFAULTS, **kw}
    trees = _plane_trees(planes)
    pairs = _adjacent_pairs(planes, near_tol=cfg["edge_on_intersection_tol"])

    for i, pl in enumerate(planes):
        verts2 = pl.get("vertices_2d") or []
        if len(verts2) < 3:
            pl["vertices_3d"], pl["edges"] = [], []
            continue
        verts3 = [[float(x), float(y), round(plane_z_at(pl, x, y), 3)] for x, y in verts2]
        pl["vertices_3d"] = verts3
        zs = [v[2] for v in verts3]
        z_min, z_max = min(zs), max(zs)

        edges = []
        n = len(verts3)
        for k in range(n):
            a, b = verts3[k], verts3[(k + 1) % n]
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length < 1e-6:
                continue
            mid_z = (a[2] + b[2]) / 2.0
            etype, exact = None, False

            for (p, q), info in pairs.items():
                if i not in (p, q):
                    continue
                other = q if i == p else p
                if max(_dist_to_line(a, info["point3"], info["dir3"]),
                       _dist_to_line(b, info["point3"], info["dir3"])) < cfg["edge_on_intersection_tol"]                         and length >= cfg["min_shared_edge_m"]:
                    exact = True
                    convex = _fold_convex(pl, planes[other],
                                          np.array([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, mid_z]),
                                          info["dir3"])
                    if convex is False:
                        etype = "u"
                    elif convex is True:
                        etype = _ridge_or_hip(pl, planes[other])
                    else:
                        etype = "h" if abs(mid_z - z_max) < 0.25 else "n"
                    break

            # odkvap: hrana na VONKAJŠOM obvode strechy — za ňou už nie sú body
            if etype is None and length >= 1.5:
                mv = np.array([-(b[1] - a[1]), (b[0] - a[0])], dtype=float)
                nl = np.linalg.norm(mv)
                if nl > 1e-9:
                    mv = mv / nl
                    M = np.array([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0])
                    cen = np.array([float(np.mean([v[0] for v in verts3])),
                                    float(np.mean([v[1] for v in verts3]))])
                    if float(np.dot(M - cen, mv)) < 0:
                        mv = -mv
                    S = M + mv * 0.7
                    nearest = [float(t.query(S)[0]) for t in trees]
                    if min(nearest) > 0.9:
                        etype = "o"

            # odkvap = najnižšia hrana A ZÁROVEŇ vodorovná (vedie po vrstevnici).
            # Stúpajúca „najnižšia" hrana je štít (audit: o4 kopíroval stúpajúcu hranu).
            _L2 = math.hypot(b[0] - a[0], b[1] - a[1])
            _slope_e = math.degrees(math.atan2(abs(b[2] - a[2]), max(_L2, 1e-9)))
            if etype is None and abs(mid_z - z_min) < 0.20 and _slope_e <= 6.0:
                etype = "o"
            if etype is None:
                etype = "s"

            edges.append({"id": f"{etype}{k+1}", "type": etype, "length_m": round(length, 3),
                          "start": a, "end": b, "exact": exact})

        pl["edges"] = _merge_edges(edges)


# ─── celý reťaz ──────────────────────────────────────────────────────────────

def build_roof_geometry(points: np.ndarray, **kw) -> List[Dict[str, Any]]:
    cfg = {**DEFAULTS, **kw}
    planes = segment_planes(points, **cfg)
    planes = split_plane_components(planes, min_points=max(80, cfg.get("min_plane_points", 150) // 2))
    planes = merge_coplanar_planes(planes, **{k: v for k, v in cfg.items() if k in ("angle_tol_deg", "dist_tol_m")})
    planes = prune_small_planes(planes, min_points=cfg.get("min_plane_points", 150))

    # Pravidelný tvar pre každý zhluk (používateľ: „ohraničiť pravidelnými tvarmi")
    for pl in planes:
        reg = regularize_polygon(pl["points"], tolerance=cfg.get("polygon_tolerance", 1.2))
        if reg:
            pl["vertices_2d"] = reg

    out = []
    for pl in planes:
        poly = pl.get("vertices_2d") or plane_polygon(pl, tolerance=cfg["polygon_tolerance"])
        if not poly:
            continue
        pl["vertices_2d"] = _shorten_ring(poly, cfg.get("min_edge_m", 1.0))
        out.append(pl)
    # iterovaný fit: ohraničenie → rovina → ohraničenie (presné priesečnice)
    refit_in_outline(out, iterations=cfg.get("refit_iterations", 0))
    for pl in out:
        reg = regularize_polygon(pl["points"], tolerance=cfg.get("polygon_tolerance", 1.2))
        if reg:
            pl["vertices_2d"] = _shorten_ring(reg, cfg.get("min_edge_m", 1.0))
    # Bez presahov: orež polygóny polrovinami overených susedov
    pairs = _adjacent_pairs(out, near_tol=max(0.5, cfg["edge_on_intersection_tol"]), min_pts=3)
    resolve_overlaps(out, pairs)
    # Topológia: spoločné rohy susedných plôch = jeden bod (žiadne medzery)
    _moved = snap_pairs_to_midpoint(out, tol_m=0.35, iterations=5)
    _closed = close_gaps_between_faces(out, pairs, tol_m=0.25, rounds=3)
    _fine = snap_pairs_to_midpoint(out, tol_m=0.12, iterations=4)
    print(f"      [topologia] snapnuté dvojice: {_moved}, dotiahnuté na hranicu suseda: {_closed}, finálny snap: {_fine}")
    from shapely.geometry import Polygon as _Poly
    for pl in out:
        v = pl.get("vertices_2d") or []
        if len(v) >= 3:
            pl["vertices_2d"] = _shorten_ring(v, cfg.get("min_edge_m", 1.0))
            pl["area_m2"] = float(_Poly(pl["vertices_2d"]).area)
    classify_edges(out, **cfg)
    return out

def clip_to_polygon(points: np.ndarray, ring_xy, buffer_m: float = 2.0) -> np.ndarray:
    """Nechá len body, ktoré padnú do obrysu (s bufferom). Obrys = autorita pôdorysu.

    Prečo: bez orezania sa do modelu dostanú susedné budovy (dôkaz: 850 m² rovín
    pri pôdoryse 344 m²). Obrys zo ZBGIS/OSM je planimetrická autorita, mračno
    dodáva výšky — presne ako hovorí rešerš (Park & Guldmann 2019).
    """
    from shapely.geometry import Point, Polygon

    pts = np.asarray(points, dtype=float)
    if len(pts) == 0 or not ring_xy:
        return pts
    poly = Polygon([(float(x), float(y)) for x, y in ring_xy])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if buffer_m:
        poly = poly.buffer(buffer_m)
    inside = np.array([poly.contains(Point(float(p[0]), float(p[1]))) for p in pts])
    return pts[inside]


def reproject_ring(ring_lonlat, target_crs: str = "EPSG:8353"):
    """Prevod obrysu z WGS84 do lokálneho CRS (S-JTSK/JTSK03)."""
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    out = []
    for lon, lat in ring_lonlat:
        x, y = t.transform(float(lon), float(lat))
        out.append([float(x), float(y)])
    return out

def clip_to_circle(points: np.ndarray, center_xy, radius_m: float) -> np.ndarray:
    """Nechá body v kruhu okolo stredu. Používa sa len ako aproximácia, keď
    geometria GIS obrysu nie je k dispozícii (sieť) — a musí sa označiť flagom."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0 or radius_m <= 0:
        return pts
    c = np.asarray(center_xy, dtype=float)
    d = np.sqrt(((pts[:, :2] - c) ** 2).sum(axis=1))
    return pts[d <= radius_m]

def intersection_edges(planes, **kw):
    """Vráti presné hrany ležiace na priesečniciach rovín (h/n/u).

    Prečo: obrys z bodov je zubatý a hrany „plávajú" ±0,3 m. Priesečnica dvoch
    rovín je presná — to je tá čiara, ktorú vidno na ortofote aj v teréne.
    Rozsah hrany berieme z bodov oboch rovín (kde sú naozaj pri sebe).
    """
    cfg = {**DEFAULTS, **kw}
    trees = _plane_trees(planes)
    pairs = _adjacent_pairs(planes, near_tol=max(0.5, cfg["edge_on_intersection_tol"]), min_pts=3)
    out = []
    for (i, j), info in pairs.items():
        point = np.asarray(info["point3"], dtype=float)
        direction = np.asarray(info["dir3"], dtype=float)
        d_xy = np.asarray(info["dir"], dtype=float)

        def t_range(pl):
            p = pl["points"]
            cross = np.abs(d_xy[0] * (p[:, 1] - info["point"][1]) - d_xy[1] * (p[:, 0] - info["point"][0]))
            sel = p[cross < cfg["edge_on_intersection_tol"]]
            if len(sel) < 5:
                return None
            t = (sel - point) @ direction
            return (float(t.min()), float(t.max()))

        t1, t2 = t_range(planes[i]), t_range(planes[j])
        if not t1 or not t2:
            continue
        t_lo, t_hi = max(t1[0], t2[0]), min(t1[1], t2[1])
        if t_hi - t_lo < cfg["min_shared_edge_m"]:
            continue

        # over pozdĺž línie, či obe roviny tam NAOZAJ majú body (inak hrana „visí")
        samples = np.linspace(t_lo, t_hi, 9)
        good = []
        for t in samples:
            q = point + direction * t
            q2 = q[:2]
            di = float(trees[i].query(q2)[0])
            dj = float(trees[j].query(q2)[0])
            good.append(di < 0.8 and dj < 0.8)
        if sum(good) >= 2:
            idxs = [k for k, g in enumerate(good) if g]
            t_lo = float(samples[min(idxs)])
            t_hi = float(samples[max(idxs)])
        if t_hi - t_lo < cfg["min_shared_edge_m"]:
            continue
        p1, p2 = point + direction * t_lo, point + direction * t_hi
        mid = (p1 + p2) / 2.0
        convex = _fold_convex(planes[i], planes[j],
                              np.array([mid[0], mid[1], (p1[2] + p2[2]) / 2.0]), direction,
                              trees=trees, idx=(i, j))
        if convex is False:
            etype = "u"
        elif convex is True:
            etype = _ridge_or_hip(planes[i], planes[j])
        else:
            etype = "n"
        out.append({
            "type": etype, "start": [float(v) for v in p1], "end": [float(v) for v in p2],
            "length_m": float(np.linalg.norm(p2 - p1)), "exact": True, "pair": [i, j],
        })
    return out

def roof_outline_edges(planes, **kw):
    """Odkvapy z obrysu CELEJ strechy (union polygónov rovín).

    Prečo: odkvap z obrysu jednotlivej roviny vedie aj vnútri strechy (vizuálna
    kontrola: „žlté línie vnútri strechy sú zbytočné"). Obrys strechy ako celku
    je zdroj pravdy pre odkvap.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    cfg = {**DEFAULTS, **kw}
    polys = []
    for pl in planes:
        v = pl.get("vertices_2d") or []
        if len(v) >= 3:
            p = Polygon(v)
            if p.is_valid and p.area > 0:
                polys.append(p)
    if not polys:
        return []
    u = unary_union(polys)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    ring = list(u.exterior.coords)
    ring = [[float(x), float(y)] for x, y in ring]

    # ktoré vrcholy obrysu ležia na priesečnici dvoch rovín (to nie je odkvap)
    pairs = _adjacent_pairs(planes, near_tol=max(0.5, cfg["edge_on_intersection_tol"]), min_pts=3)
    out = []
    for i in range(len(ring) - 1):
        a, b = ring[i], ring[i + 1]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < cfg.get("min_edge_m", 1.0):
            continue
        def _d2(pt, info):
            d = np.asarray(info["dir"], dtype=float)
            p0 = np.asarray(info["point"], dtype=float)
            return abs(d[0] * (pt[1] - p0[1]) - d[1] * (pt[0] - p0[0]))

        on_int = False
        for info in pairs.values():
            if max(_d2(a, info), _d2(b, info)) < cfg["edge_on_intersection_tol"]:
                on_int = True
                break
        if on_int:
            continue
        # nájdi rovinu, ktorá tam leží (pre výšky v OBIDVoch koncoch)
        owner = None
        from shapely.geometry import Point as _Pt
        for pl in planes:
            if Polygon(pl["vertices_2d"]).distance(_Pt(a[0], a[1])) < 0.05 or \
               Polygon(pl["vertices_2d"]).distance(_Pt(b[0], b[1])) < 0.05:
                owner = pl
                break
        if owner is None:
            continue
        z1 = plane_z_at(owner, a[0], a[1])
        z2 = plane_z_at(owner, b[0], b[1])
        # odkvap vedie po vrstevnici → musí byť takmer vodorovný; inak je to štít
        slope_deg = math.degrees(math.atan2(abs(z2 - z1), max(L, 1e-9)))
        etype = "o" if slope_deg <= 6.0 else "s"
        out.append({"type": etype, "start": [a[0], a[1], round(z1, 3)], "end": [b[0], b[1], round(z2, 3)],
                    "length_m": round(L, 3), "exact": False})
    return out

def planes_from_masks(points, masks, min_points: int = 120, min_area_m2: float = 4.0,
                      shrink_m: float = 0.25):
    """Roviny riadené modelom: obrys z masky modelu, sklon/výška z mračna.

    Princíp z rešerše: „hrany z obrazu, roviny z mračna". Maska určí, KDE plocha je
    (presná hranica z modelu), mračno určí JEJ sklon a výšku.
    """
    import numpy as _np
    from matplotlib.path import Path as MplPath
    from shapely.geometry import Polygon

    out = []
    for mk in masks:
        ring = mk.get("polygon_xy") or []
        if len(ring) < 3:
            continue
        poly = Polygon(ring).buffer(0)
        if poly.area < min_area_m2 or poly.is_empty:
            continue
        inner = poly.buffer(-shrink_m) if shrink_m else poly
        if inner.is_empty:
            inner = poly
        path = MplPath(_np.array(inner.exterior.coords))
        sel = path.contains_points(_np.asarray(points, dtype=float)[:, :2])
        sub = np.asarray(points, dtype=float)[sel]
        if len(sub) < min_points:
            continue                      # v maske nie je dosť bodov → plocha sa preskočí
        nv, d = _fit_plane_svd(sub)
        dist = np.abs(sub @ nv + d)
        out.append({
            "normal": nv, "d": float(d), "points": sub, "n_points": int(len(sub)),
            "rmse_m": round(float(np.sqrt((dist ** 2).mean())), 4),
            "slope_deg": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
            "azimuth_deg": round((math.degrees(math.atan2(nv[0], nv[1])) + 360.0) % 360.0, 1),
            "area_m2": float(poly.area),
            "vertices_2d": [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]],
            "mask_class": mk.get("class"),
            "source": "vision_mask",
        })
    return out


def build_roof_geometry_from_masks(points, masks, **kw):
    """Celá geometria riadená modelom: masky → roviny → orezanie → hrany."""
    from shapely.geometry import Polygon as _P

    cfg = {**DEFAULTS, **kw}
    planes = planes_from_masks(points, masks, min_points=cfg.get("min_plane_points", 150))
    if not planes:
        return []
    pairs = _adjacent_pairs(planes, near_tol=cfg["edge_on_intersection_tol"])
    resolve_overlaps(planes, pairs)
    snap_vertices_to_intersections(planes, pairs, tol_m=0.9)
    for pl in planes:
        v = pl.get("vertices_2d") or []
        if len(v) >= 3:
            pl["vertices_2d"] = _shorten_ring(v, cfg.get("min_edge_m", 1.0))
            pl["area_m2"] = float(_P(pl["vertices_2d"]).area)
    # pri maskách je hranica z modelu o niečo hrubšia → väčšia tolerancia pre hrany
    classify_edges(planes, **{**cfg, "edge_on_intersection_tol": 0.60})
    return planes

def snap_vertices_to_intersections(planes, pairs, tol_m: float = 0.9):
    """Prisunie vrcholy obrysu na priesečnicu susedných rovín.

    Hranica masky z modelu je presná len na ~0,3–0,6 m (rozlišenie modelu),
    kým priesečnica dvoch rovín je presná. Po snapnutí leží hrana presne na
    priesečnici → dá sa klasifikovať ako hrebeň/nárožie/úžľabie.
    """
    for (i, j), info in pairs.items():
        p0 = np.asarray(info["point"], dtype=float)
        d = np.asarray(info["dir"], dtype=float)
        ln = np.linalg.norm(d)
        if ln < 1e-9:
            continue
        d = d / ln
        for idx in (i, j):
            verts = planes[idx].get("vertices_2d") or []
            if len(verts) < 3:
                continue
            new = []
            for v in verts:
                q = np.asarray(v, dtype=float)
                t = float((q - p0) @ d)
                proj = p0 + d * t
                dist = float(np.linalg.norm(q - proj))
                if dist <= tol_m and abs(t) <= 20.0:
                    new.append([float(proj[0]), float(proj[1])])
                else:
                    new.append([float(q[0]), float(q[1])])
            planes[idx]["vertices_2d"] = new

def register_offset(liDAR_ring, mask_rings, search_m: float = 3.0, step_m: float = 0.25):
    """Deterministická co-registrácia: nájdi posun masiek, ktorý maximalizuje IoU.

    Prečo: masky z modelu a mračno majú spoločný pôvod len približne (merané IoU 0,58);
    bez tohto kroku padajú fity rovín do zlých sklonov (namerané 46–64° namiesto ~30°).
    """
    from shapely.affinity import translate
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if not liDAR_ring or not mask_rings:
        return 0.0, 0.0, 0.0
    lid = Polygon(liDAR_ring).buffer(0)
    polys = [Polygon(r).buffer(0) for r in mask_rings if len(r) >= 3]
    if not polys:
        return 0.0, 0.0, 0.0
    base = unary_union(polys)

    best, best_iou = (0.0, 0.0), -1.0
    n = int(search_m / step_m)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            dx, dy = i * step_m, j * step_m
            moved = translate(base, xoff=dx, yoff=dy)
            inter = lid.intersection(moved).area
            uni = lid.union(moved).area
            iou = inter / uni if uni > 0 else 0.0
            if iou > best_iou:
                best_iou, best = iou, (dx, dy)
    return best[0], best[1], best_iou

def regularize_polygon(points, tolerance: float = 1.0, rect_tol: float = 1.10, tri_tol: float = 1.10):
    """Ohraničí zhluk bodov pravidelným tvarom, keď sa to hodí.

    Poradie: minimálny rotovaný obdĺžnik → minimálny trojuholník → (inak) vyhladený obrys.
    Vďaka tomu má plocha 3–4 rohy namiesto desiatok zubatých vrcholov a hrany
    (odkvap, nárožie, hrebeň, úžľabie) vznikajú ako spoločné línie týchto tvarov.
    """
    import cv2
    from shapely.geometry import MultiPoint, Polygon

    pts = np.asarray(points, dtype=float)[:, :2]
    if len(pts) < 4:
        return None
    # referenciou je KONKÁVNY obal (konvexný obal rozptýlených bodov nafukuje plochu)
    from shapely import concave_hull as _ch

    mp = MultiPoint([(float(x), float(y)) for x, y in pts])
    hull = None
    for ratio in (0.25, 0.4):
        try:
            c = _ch(mp, ratio=ratio)
            if c.geom_type == "Polygon" and c.area > 0:
                hull = c
                break
        except Exception:
            continue
    if hull is None:
        hull = mp.convex_hull
    hull_area = float(hull.area)
    if hull_area <= 0:
        return None

    p32 = pts.astype(np.float32).reshape(-1, 1, 2)
    best = None

    try:
        rect = cv2.minAreaRect(p32)
        rpts = cv2.boxPoints(rect)
        rp = Polygon([(float(x), float(y)) for x, y in rpts])
        if rp.is_valid and 0 < rp.area <= hull_area * rect_tol:
            best = rp
    except Exception:
        pass

    if best is None:
        try:
            _, tri = cv2.minEnclosingTriangle(p32)
            tp = Polygon([(float(a), float(b)) for a, b in tri.reshape(-1, 2)])
            if tp.is_valid and 0 < tp.area <= hull_area * tri_tol:
                best = tp
        except Exception:
            pass

    if best is None:
        best = hull

    best = best.simplify(tolerance, preserve_topology=True)
    if best.geom_type != "Polygon" or best.is_empty:
        return None
    ring = [[float(x), float(y)] for x, y in best.exterior.coords[:-1]]
    return ring if len(ring) >= 3 else None

def refit_in_outline(planes, iterations: int = 1, margin_m: float = 0.15):
    """Iteruje: ohraničenie → fit roviny len na body vo vnútri → nové ohraničenie.

    Prečo: pôvodný fit používa všetky inlier body (aj tie, čo patria susednej ploche),
    preto sú priesečnice posunuté o 0,5–1,4 m (merané voči skutočným líniám v ortofote).
    Ohraničenie je „tvar plochy", takže fit je čistý a hrany sedia.
    """
    from matplotlib.path import Path as MplPath
    from shapely.geometry import Polygon

    for _ in range(max(0, iterations)):
        for pl in planes:
            ring = pl.get("vertices_2d") or []
            if len(ring) < 3:
                continue
            poly = Polygon(ring).buffer(0)
            if poly.is_empty:
                continue
            inner = poly.buffer(-margin_m)
            if inner.is_empty or inner.geom_type != "Polygon":
                inner = poly
            path = MplPath(np.array(inner.exterior.coords))
            sel = path.contains_points(pl["points"][:, :2])
            sub = pl["points"][sel]
            if len(sub) < 150:                # primalo bodov → ponechaj pôvodný fit
                continue
            nv, d = _fit_plane_svd(sub)
            dist = np.abs(sub @ nv + d)
            pl.update({
                "normal": nv, "d": float(d), "points": sub, "n_points": int(len(sub)),
                "rmse_m": round(float(np.sqrt((dist ** 2).mean())), 4),
                "slope_deg": round(math.degrees(math.acos(min(1.0, abs(float(nv[2]))))), 2),
                "azimuth_deg": round((math.degrees(math.atan2(nv[0], nv[1])) + 360.0) % 360.0, 1),
            })
    return planes

def snap_shared_vertices(planes, tol_m: float = 0.30):
    """Spoločné vrcholy susedných plôch prisunie na jeden bod (medoid zhluku).

    Výsledok: susedné plochy zdieľajú presné rohy → žiadne medzery (AGENTS.md),
    a hrany tak tvoria uzavretú sieť.
    """
    from collections import defaultdict

    from scipy.spatial import cKDTree

    pts = []                                   # (index_roviny, index_vrcholu, x, y)
    for k, pl in enumerate(planes):
        for m, v in enumerate(pl.get("vertices_2d") or []):
            pts.append((k, m, float(v[0]), float(v[1])))
    if len(pts) < 2:
        return 0

    arr = np.array([[p[2], p[3]] for p in pts])
    tree = cKDTree(arr)
    # len vrcholy z ROZDIELNYCH plôch (v rámci jednej plochy sa nesnapuje!)
    # Greedové párové zlučovanie: zhluk max 4, každý vrchol z inej roviny, najbližšie prvé.
    # (union-find bez guardov reťazil 57 vrcholov a plochy sa zrútili na 0 m²)
    close = sorted(tree.query_pairs(tol_m),
                   key=lambda ab: float(np.linalg.norm(arr[ab[0]] - arr[ab[1]])))
    parent = list(range(len(arr)))
    cluster = {i: [i] for i in range(len(arr))}
    planes_in = {i: {pts[i][0]} for i in range(len(arr))}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if len(cluster[ra]) + len(cluster[rb]) > 4:
            return
        if planes_in[ra] & planes_in[rb]:
            return
        parent[rb] = ra
        cluster[ra] = cluster[ra] + cluster[rb]
        planes_in[ra] |= planes_in[rb]

    for a, b in close:
        union(a, b)

    newpos = arr.copy()
    moved = 0
    for root, mem in list(cluster.items()):
        if len(mem) < 2:
            continue
        sub = arr[mem]
        c = sub.mean(axis=0)
        d = np.linalg.norm(sub - c, axis=1)
        pos = sub[int(np.argmin(d))]
        for i in mem:
            if abs(newpos[i][0] - pos[0]) > 1e-9 or abs(newpos[i][1] - pos[1]) > 1e-9:
                moved += 1
            newpos[i] = pos

    for n, (k, m, _, _) in enumerate(pts):
        vv = planes[k]["vertices_2d"]
        vv[m][0], vv[m][1] = float(newpos[n][0]), float(newpos[n][1])
    return moved


def eave_loop_summary(edges_o) -> List[Dict[str, Any]]:
    """Spojí úseky odkvapus do slučiek a vráti ich súhrn (uzavretá? dĺžka?)."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge

    if not edges_o:
        return []
    seg = [[float(e["start"][0]), float(e["start"][1]), float(e["end"][0]), float(e["end"][1])]
           for e in edges_o if e.get("type") == "o" and e.get("length_m", 0) >= 1.0]
    if not seg:
        return []
    # spojiť blízke konce (do 1,0 m) — inak obvod zostane rozpadnutý
    for _ in range(3):
        for i in range(len(seg)):
            for j in range(len(seg)):
                if i == j:
                    continue
                best = None
                for a in (0, 2):
                    for b in (0, 2):
                        d = math.hypot(seg[i][a] - seg[j][b], seg[i][a + 1] - seg[j][b + 1])
                        if best is None or d < best[0]:
                            best = (d, a, b)
                d, a, b = best
                if 0 < d <= 1.0:
                    mx = (seg[i][a] + seg[j][b]) / 2.0
                    my = (seg[i][a + 1] + seg[j][b + 1]) / 2.0
                    seg[i][a], seg[i][a + 1] = mx, my
                    seg[j][b], seg[j][b + 1] = mx, my
    lines = [LineString([(s[0], s[1]), (s[2], s[3])]) for s in seg]
    merged = linemerge(lines)
    geoms = list(merged.geoms) if merged.geom_type == "MultiLineString" else [merged]
    loops = []
    for g in geoms:
        ring = list(g.coords)
        closed = math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) < 0.25
        loops.append({"closed": bool(closed), "length_m": round(float(g.length), 2),
                      "points": len(ring)})
    return loops

def snap_pairs_to_midpoint(planes, tol_m: float = 0.25, iterations: int = 4):
    """Susedné vrcholy rôznych plôch posúva na ich stred — medzery sa zmenšujú na polovicu
    každou iteráciou (po 4 ≈ tol/16 < 5 cm), bez reťazenia a bez kolapsu plôch.

    Každá iterácia: každý vrchol sa zúčastní najviac jedného páru (a rovnaký partner tiež),
    takže sa neposúvajú dva vrcholy jednej plochy na ten istý bod.
    """
    from scipy.spatial import cKDTree

    total = 0
    for _ in range(max(1, iterations)):
        pts = []
        for k, pl in enumerate(planes):
            for m, v in enumerate(pl.get("vertices_2d") or []):
                pts.append((k, m, float(v[0]), float(v[1])))
        if len(pts) < 2:
            return total
        arr = np.array([[p[2], p[3]] for p in pts])
        tree = cKDTree(arr)

        used = set()
        moved = 0
        for i in range(len(pts)):
            if i in used:
                continue
            dists, idxs = tree.query(arr[i], k=min(8, len(pts)))
            for dist, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                j = int(j)
                if j == i or j in used or pts[j][0] == pts[i][0]:
                    continue
                if float(dist) <= tol_m:
                    mx = (pts[i][2] + pts[j][2]) / 2.0
                    my = (pts[i][3] + pts[j][3]) / 2.0
                    for (k, m, _, _) in (pts[i], pts[j]):
                        planes[k]["vertices_2d"][m][0] = float(mx)
                        planes[k]["vertices_2d"][m][1] = float(my)
                    used.add(i)
                    used.add(j)
                    moved += 1
                break
        total += moved
        if moved == 0:
            break
    return total

def detect_roof_features(planes, min_chimney_points: int = 12, min_panel_points: int = 30):
    """Nájde komíny a potenciálne solárne panely (zvyšky/odskoky nad rovinou).

    Vracia (chimneys, panels) — každý prvok {plane, x, y, z, area_m2, points, kind}.
    """
    from scipy import ndimage

    chimneys, panels = [], []
    for i, pl in enumerate(planes):
        pts = pl["points"]
        if len(pts) < 50:
            continue
        n = pl["normal"]
        d = pl["d"]
        dist = pts @ n + d                       # kladné = nad rovinou
        for lo, hi, kind, minpts, minA, maxA, out in (
            (0.20, 3.0, "komin", 8, 0.25, 10.0, chimneys),
            (0.08, 0.35, "panel", min_panel_points, 2.0, 25.0, panels),
        ):
            sel = (dist > lo) & (dist < hi)
            sub = pts[sel]
            if len(sub) < minpts:
                continue
            cell = 0.25
            mins = sub[:, :2].min(axis=0) - cell
            idx = np.floor((sub[:, :2] - mins) / cell).astype(np.int64)
            shape = tuple((idx.max(axis=0) + 3).tolist())
            grid = np.zeros(shape, dtype=bool)
            grid[idx[:, 0], idx[:, 1]] = True
            lab, k = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
            lb = lab[idx[:, 0], idx[:, 1]]
            for c in range(1, k + 1):
                m = lb == c
                cnt = int(m.sum())
                if cnt < minpts:
                    continue
                sub2 = sub[m]
                area = float(cnt * cell * cell)
                if not (minA <= area <= maxA):
                    continue
                out.append({
                    "plane": i, "kind": kind,
                    "x": float(sub2[:, 0].mean()), "y": float(sub2[:, 1].mean()),
                    "z": float(sub2[:, 2].mean()),
                    "area_m2": round(area, 2), "points": cnt,
                })
    # strmé malé plochy (komínové telo) — doplň medzi komíny
    for i, pl in enumerate(planes):
        if pl.get("slope_deg", 0) > 45 and 1.5 <= (pl.get("area_m2") or 0) <= 10.0:
            chimneys.append({"plane": i, "kind": "komin_strecha", "x": 0.0, "y": 0.0,
                             "z": 0.0, "area_m2": round(float(pl["area_m2"]), 2),
                             "points": int(pl.get("n_points") or 0)})
    return chimneys, panels

def roof_outline_ring(planes, tolerance: float = 0.6):
    """Vonkajší obrys strechy (exterior zjednotenia plôch) — uzavretý obvod odkvapu."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polys = []
    for pl in planes:
        v = pl.get("vertices_2d") or []
        if len(v) >= 3:
            p = Polygon(v)
            if p.is_valid and p.area > 0:
                polys.append(p)
    if not polys:
        return []
    u = unary_union(polys)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda g: g.area)
    ring = u.simplify(tolerance).exterior
    return [[float(x), float(y)] for x, y in ring.coords]


def detect_features_from_cloud(planes, all_points, min_chimney_points: int = 8):
    """Komíny a panely z celého mračna: body 0,2–3,0 m (komín) / 0,08–0,35 m (panel)
    nad najbližšou rovinou a v jej pôdoryse."""
    from matplotlib.path import Path as MplPath
    from scipy import ndimage

    pts = np.asarray(all_points, dtype=float)
    if len(pts) == 0 or not planes:
        return [], []
    normals = np.stack([p["normal"] for p in planes])
    ds = np.array([p["d"] for p in planes])
    dist = np.abs(pts @ normals.T + ds)          # (N, K)
    nearest = np.argmin(dist, axis=1)
    dmin = dist[np.arange(len(pts)), nearest]

    paths = []
    for pl in planes:
        v = pl.get("vertices_2d") or []
        paths.append(MplPath(np.array(v)) if len(v) >= 3 else None)

    inside = np.zeros(len(pts), dtype=bool)
    for k, path in enumerate(paths):
        m = nearest == k
        if path is None or not m.any():
            continue
        inside[m] = path.contains_points(pts[m, :2])

    def cluster(lo, hi, minpts, minA, maxA):
        sel = inside & (dmin > lo) & (dmin < hi)
        sub = pts[sel]
        out = []
        if len(sub) < minpts:
            return out
        cell = 0.25
        mins = sub[:, :2].min(axis=0) - cell
        idx = np.floor((sub[:, :2] - mins) / cell).astype(np.int64)
        grid = np.zeros(tuple((idx.max(axis=0) + 3).tolist()), dtype=bool)
        grid[idx[:, 0], idx[:, 1]] = True
        lab, k = ndimage.label(grid, structure=np.ones((3, 3), dtype=int))
        lb = lab[idx[:, 0], idx[:, 1]]
        for c in range(1, k + 1):
            m = lb == c
            cnt = int(m.sum())
            if cnt < minpts:
                continue
            area = float(cnt * cell * cell)
            if not (minA <= area <= maxA):
                continue
            s2 = sub[m]
            out.append({"x": float(s2[:, 0].mean()), "y": float(s2[:, 1].mean()),
                        "z": float(s2[:, 2].mean()), "area_m2": round(area, 2), "points": cnt})
        return out

    chimneys = cluster(0.20, 3.0, min_chimney_points, 0.25, 10.0)
    panels = cluster(0.08, 0.35, 30, 2.0, 25.0)
    return chimneys, panels

def _support_lines_from_hull(pts2d, min_seg_m: float = 1.2, angle_tol_deg: float = 22.0):
    """Nosné línie vonkajšieho obrysu: z konvexného obalu, krátke a takmer rovnobežné
    úseky sa spájajú do jednej línie (strešná hrana je rovná línia, nie zubatá)."""
    from shapely.geometry import MultiPoint

    hull = MultiPoint([(float(x), float(y)) for x, y in pts2d]).convex_hull
    coords = list(hull.exterior.coords)
    segs = []
    for k in range(len(coords) - 1):
        a = np.array(coords[k], dtype=float)
        b = np.array(coords[k + 1], dtype=float)
        L = float(np.linalg.norm(b - a))
        if L < min_seg_m:
            continue
        segs.append({"a": a, "b": b, "d": (b - a) / L, "L": L})

    lines = []
    for s in segs:
        ang = math.degrees(math.atan2(s["d"][1], s["d"][0])) % 180.0
        placed = False
        for ln in lines:
            diff = abs(ang - ln["ang"])
            diff = min(diff, 180.0 - diff)
            if diff <= angle_tol_deg:
                # je to tá istá línia? (vzdialenosť stredu úseku od línie)
                m = (s["a"] + s["b"]) / 2.0
                dist = abs(ln["d"][0] * (m[1] - ln["p"][1]) - ln["d"][1] * (m[0] - ln["p"][0]))
                if dist <= 1.0:
                    ln["L"] += s["L"]
                    placed = True
                    break
        if not placed:
            lines.append({"p": s["a"], "d": s["d"], "L": s["L"], "ang": ang})
    lines.sort(key=lambda l: -l["L"])
    return lines[:6]


def faces_from_halfplanes(planes, pairs, cfg=None):
    """Pre každú rovinu vráti obrys ako prienik polrovín (presné rohy)."""
    from shapely.geometry import MultiPoint, Polygon

    cfg = cfg or {}
    big = cfg.get("halfplane_margin", 4.0)
    for i, pl in enumerate(planes):
        pts = pl["points"][:, :2]
        if len(pts) < 30:
            continue
        cen = pts.mean(axis=0)
        minx, miny = pts.min(axis=0) - big
        maxx, maxy = pts.max(axis=0) + big
        poly = [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]]

        # 1) polroviny od susedov (spoločná čiara pre obe plochy → spoločná hrana)
        for (ii, jj), info in pairs.items():
            if i not in (ii, jj):
                continue
            d = np.asarray(info["dir"], dtype=float)
            p0 = np.asarray(info["point"], dtype=float)
            side = d[0] * (cen[1] - p0[1]) - d[1] * (cen[0] - p0[0])
            keep = 1 if side >= 0 else -1
            clipped = _clip_polygon_by_line(poly, p0, d, keep)
            if clipped:
                poly = clipped

        # 2) polroviny vonkajších hrán (nosné línie obrysu)
        for ln in _support_lines_from_hull(pts):
            p0 = ln["p"]
            d = ln["d"]
            side = d[0] * (cen[1] - p0[1]) - d[1] * (cen[0] - p0[0])
            keep = 1 if side >= 0 else -1
            clipped = _clip_polygon_by_line(poly, p0, d, keep)
            if clipped:
                poly = clipped

        if len(poly) < 3:
            continue
        p = Polygon(poly)
        if not p.is_valid or p.area < 1.0:
            continue
        # poistka len proti hrubému vybehnutiu (bez nej by plochy vznikali mimo strechy);
        # POZOR: príliš tesný orez vytváral medzery medzi plochami (únia sa rozpadla)
        hull = MultiPoint([(float(x), float(y)) for x, y in pts]).convex_hull.buffer(2.0)
        inter = p.intersection(hull)
        if not inter.is_empty and inter.geom_type == "Polygon" and inter.area >= 0.7 * p.area:
            p = inter
        if p.is_empty or p.geom_type != "Polygon" or p.area < 1.0:
            continue
        pl["vertices_2d"] = [[float(x), float(y)] for x, y in list(p.exterior.coords)[:-1]]
        pl["area_m2"] = float(p.area)

def close_gaps_between_faces(planes, pairs, tol_m: float = 0.25, rounds: int = 3):
    """Vrcholy blízko hranice suseda sa premietnu PRESNE na tú hranicu.

    Výsledok: susedné plochy zdieľajú tú istú hranu (únia je súvislá), presne ako
    na skutočnej streche — bez medzier a bez presahov (AGENTS.md).
    """
    from shapely.geometry import Point, Polygon

    moved = 0
    for _ in range(max(1, rounds)):
        touched = 0
        for (i, j) in pairs:
            for own, other in ((i, j), (j, i)):
                pv = planes[other].get("vertices_2d")
                if not pv or len(pv) < 3:
                    continue
                boundary = Polygon(pv).buffer(0).boundary
                cur = planes[own].get("vertices_2d") or []
                new = []
                for v in cur:
                    p = Point(float(v[0]), float(v[1]))
                    d = p.distance(boundary)
                    if 0 < d <= tol_m:
                        q = boundary.interpolate(boundary.project(p))
                        new.append([float(q.x), float(q.y)])
                        touched += 1
                    else:
                        new.append([float(v[0]), float(v[1])])
                planes[own]["vertices_2d"] = new
        moved += touched
        if touched == 0:
            break
    return moved
