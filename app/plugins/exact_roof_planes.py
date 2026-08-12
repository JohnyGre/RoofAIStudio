"""
exact_roof_planes.py
=====================

Nahrádza fragmentovanú (mnoho "f" hrán) rekonštrukciu z `_classify_edges` +
`_fix_roof_topology` presnou geometrickou metódou:

    1. RANSAC/DBSCAN segmentácia rovín (rovnaké ako trimesh_roof_splitter)
    2. Vyčistenie hranice každej roviny (convex hull + cv2.approxPolyDP)
    3. OVERENÁ susednosť dvoch rovín = veľa bodov OBOCH mrakov leží blízko
       PRESNÉHO priesečníka ich dvoch nekonečných rovinových rovníc
       (nie len "hranice sú blízko seba" - to je krehké, viď pôvodný kód)
    4. Orezanie polygónu roviny polrovinou (Sutherland-Hodgman) oproti
       KAŽDÉMU overenému susedovi - výsledná hrana je EXAKTNY priesečník,
       nie odhad/snap
    5. Klasifikácia hrany (hreben/narozie/uzlabie) cez smerovú deriváciu
       výšky (skutočný dihedrálny uhol, nie heuristika)

Výstup je v ROVNAKOM formáte ako `trimesh_roof_splitter.mesh_to_roof_planes`,
takže je to drop-in náhrada pre `viewer_generator.py`.

POUŽITIE:
    from app.plugins.exact_roof_planes import mesh_to_roof_planes_exact
    planes = mesh_to_roof_planes_exact("/cesta/k/scan.obj")
    # planes -> rovnaky format ako predtym, das do viewer_generator.py
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger("Roof AI Studio")


# ---------------------------------------------------------------------------
# Geometria - RANSAC + presné orezanie
# ---------------------------------------------------------------------------

def _fit_plane_svd(points: np.ndarray):
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    d = -normal.dot(centroid)
    return normal, d, centroid


def _ransac_plane(points, distance_threshold=0.06, num_iterations=600, min_inliers=150, rng=None):
    rng = rng or np.random.default_rng(42)
    n = len(points)
    best_inliers = np.array([], dtype=int)
    best_normal, best_d = None, None
    for _ in range(num_iterations):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len
        d = -normal.dot(p0)
        dist = np.abs(points @ normal + d)
        inliers = np.where(dist < distance_threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal, best_d = normal, d
    if len(best_inliers) < min_inliers:
        return None, None, best_inliers
    normal, d, _ = _fit_plane_svd(points[best_inliers])
    dist = np.abs(points @ normal + d)
    inliers = np.where(dist < distance_threshold)[0]
    return normal, d, inliers


def _multi_plane_ransac(points, distance_threshold=0.06, min_inliers=200, max_planes=15,
                         min_normal_z=0.15, density_outlier_factor=3.0,
                         flat_roof_max_degree=8.0):
    """Multi-plane RANSAC s filtrom na 'strechovite' roviny + odfiltrovaním
    podozrivo riedkych (nizka hustota bodov = pravdepodobne zla) rovin."""
    remaining_idx = np.arange(len(points))
    raw_planes = []
    pts = points.copy()
    while len(remaining_idx) > min_inliers and len(raw_planes) < max_planes:
        normal, d, inl_local = _ransac_plane(pts[remaining_idx], distance_threshold, 600, min_inliers)
        if normal is None:
            break
        if abs(normal[2]) < min_normal_z:
            remaining_idx = np.delete(remaining_idx, inl_local)
            continue
        global_idx = remaining_idx[inl_local]
        raw_planes.append({"normal": normal, "d": d, "point_idx": global_idx})
        remaining_idx = np.delete(remaining_idx, inl_local)


    # --- Connected components split ---
    # RANSAC moze zlepit 2 priestorovo oddelene plochy s rovnakym sklonom
    # (napr. dve rovne casti strechy na opacnych stranach budovy).
    # Ak rovina obsahuje 2+ signifikantne zhluky, rozdel ju na samostatne roviny.
    SPLIT_MIN_FRACTION = 0.05  # min 5% bodov zhluku musi byt, aby bol signifikantny
    SPLIT_MIN_POINTS = 50
    split_planes = []
    for p in raw_planes:
        pts3d = points[p["point_idx"]]
        n = len(pts3d)
        tree = cKDTree(pts3d)
        dists, _ = tree.query(pts3d, k=min(n, 2))
        if n < 2:
            split_planes.append(p)
            continue
        typical_spacing = float(np.median(dists[:, 1]))
        radius = typical_spacing * 3.0
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            split_planes.append(p)
            continue
        row, col = pairs[:, 0], pairs[:, 1]
        graph = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        n_comp, labels = connected_components(graph, directed=False)
        if n_comp <= 1:
            split_planes.append(p)
            continue
        # Rozdel na samostatne roviny
        sizes = [(int(np.sum(labels == c)), c) for c in range(n_comp)]
        sizes.sort(reverse=True)
        significant = [(sz, c) for sz, c in sizes if sz >= SPLIT_MIN_POINTS and sz >= n * SPLIT_MIN_FRACTION]
        if len(significant) <= 1:
            # Iba jeden velky zhluk + noise -> nechaj ako je
            split_planes.append(p)
            continue
        logger.info(
            "exact_roof_planes: rovina rozdelena na %d suvisle casti (rozdelene na samostatne roviny)",
            len(significant),
        )
        for sz, c in significant:
            new_p = dict(p)
            new_p["point_idx"] = p["point_idx"][labels == c]
            new_p["_split_from"] = True
            split_planes.append(new_p)
    raw_planes = split_planes


    # over hustotu bodov (plocha_hull / n_bodov) - vyrad podozrive riedke roviny
    densities = []
    for p in raw_planes:
        pts3d = points[p["point_idx"]]
        area = _hull_area_approx(pts3d, p["normal"])
        density = area / max(len(p["point_idx"]), 1)
        densities.append(density)
        p["_density"] = density

    # oznac ploche strechy (sklon < flat_roof_max_degree) - nie low_confidence
    # Ploche strechy maju prirodzene vyssiu density (menej bodov na vacsej ploche
    # kvoly kolmemu dopadu lasera na takmer vodorovnu plochu)
    for p in raw_planes:
        slope_deg = float(np.degrees(np.arccos(np.clip(abs(p["normal"][2]), -1, 1))))
        p["_is_flat"] = slope_deg < flat_roof_max_degree

    if densities:
        med = float(np.median(densities))
        for p in raw_planes:
            if not p.get("_is_flat"):
                p["_low_confidence"] = p["_density"] > med * density_outlier_factor

    return raw_planes


def _hull_area_approx(pts3d, normal):
    u, v = _plane_basis(normal)
    centroid = pts3d.mean(axis=0)
    local2d = np.column_stack([(pts3d - centroid) @ u, (pts3d - centroid) @ v])
    try:
        return float(ConvexHull(local2d).volume)
    except Exception:
        return 0.0


def _plane_basis(normal):
    tmp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def _clean_boundary_polygon(pts3d, normal, epsilon_ratio=0.02):
    """Convex hull + cv2.approxPolyDP -> cisty polygon (malo vrcholov)."""
    u, v = _plane_basis(normal)
    centroid = pts3d.mean(axis=0)
    local2d = np.column_stack([(pts3d - centroid) @ u, (pts3d - centroid) @ v]).astype(np.float32)
    hull = ConvexHull(local2d)
    hull_local = local2d[hull.vertices]
    contour = hull_local.reshape(-1, 1, 2)
    perim = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perim
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    poly3d = centroid + np.outer(simplified[:, 0], u) + np.outer(simplified[:, 1], v)
    return [tuple(p) for p in poly3d]


def _polygon_area_3d(poly, normal) -> float:
    """Plocha planarneho 3D polygonu (shoelace v lokalnych 2D suradniciach roviny)."""
    if len(poly) < 3:
        return 0.0
    u, v = _plane_basis(np.asarray(normal))
    poly_arr = np.asarray(poly)
    c = poly_arr.mean(axis=0)
    x = (poly_arr - c) @ u
    y = (poly_arr - c) @ v
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def clip_polygon_by_plane(polygon, plane_point, plane_normal, keep_positive=True):
    """Sutherland-Hodgman: orez planarny 3D polygon polrovinou danej rovinou.
    Cisty Python ekvivalent Blenderovho bmesh.ops.bisect_plane (bez Blenderu)."""
    plane_point = np.asarray(plane_point, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)

    def side(p):
        s = np.dot(np.asarray(p) - plane_point, plane_normal)
        return s if keep_positive else -s

    output = []
    n = len(polygon)
    if n == 0:
        return []
    for i in range(n):
        cur, nxt = np.asarray(polygon[i]), np.asarray(polygon[(i + 1) % n])
        s_cur, s_nxt = side(cur), side(nxt)
        if s_cur >= -1e-9:
            output.append(tuple(cur))
        if (s_cur >= -1e-9) != (s_nxt >= -1e-9):
            t = s_cur / (s_cur - s_nxt)
            inter = cur + t * (nxt - cur)
            output.append(tuple(inter))
    return output


def _confirmed_adjacent_planes(points, raw_planes, near_tol=0.50, min_near_points=8, min_length_m=2.0):
    """Over susednost cez presny priesecnik rovin + overenie bodmi oboch mrakov."""
    pairs = {}
    n_planes = len(raw_planes)
    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            n1, d1 = raw_planes[i]["normal"], raw_planes[i]["d"]
            n2, d2 = raw_planes[j]["normal"], raw_planes[j]["d"]
            direction = np.cross(n1, n2)
            dnorm = np.linalg.norm(direction)
            if dnorm < 1e-8:
                continue
            direction = direction / dnorm
            A = np.array([n1, n2, direction])
            b = np.array([-d1, -d2, 0.0])
            try:
                point = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue

            pts_i = points[raw_planes[i]["point_idx"]]
            pts_j = points[raw_planes[j]["point_idx"]]

            def dist_and_t(pts):
                rel = pts - point
                t = rel @ direction
                perp = rel - np.outer(t, direction)
                return np.linalg.norm(perp, axis=1), t

            dist_i, t_i = dist_and_t(pts_i)
            dist_j, t_j = dist_and_t(pts_j)
            near_i = t_i[dist_i < near_tol]
            near_j = t_j[dist_j < near_tol]
            if len(near_i) < min_near_points or len(near_j) < min_near_points:
                continue

            t_lo, t_hi = max(near_i.min(), near_j.min()), min(near_i.max(), near_j.max())
            if t_hi - t_lo < min_length_m:
                continue

            pairs[(i, j)] = {"point": point, "direction": direction, "t_lo": t_lo, "t_hi": t_hi}
    return pairs


def _dihedral_type(pair_info, plane_a, plane_b) -> Tuple[str, float]:
    """Konvexne (hreben/narozie) vs konkavne (uzlabie) cez smerovu derivaciu vysky."""
    p_start = pair_info["point"] + pair_info["t_lo"] * pair_info["direction"]
    p_end = pair_info["point"] + pair_info["t_hi"] * pair_info["direction"]
    mid = (p_start + p_end) / 2.0
    z_var = abs(p_start[2] - p_end[2])

    def f(plane, x, y):
        n, d = plane["normal"], plane["d"]
        return -(n[0] * x + n[1] * y + d) / n[2]

    def dz_away(plane, cx, cy, x0, y0, z0):
        dirx, diry = cx - x0, cy - y0
        nrm = math.hypot(dirx, diry)
        if nrm < 1e-6:
            return 0.0
        dirx, diry = dirx / nrm, diry / nrm
        return f(plane, x0 + dirx, y0 + diry) - z0

    ca = plane_a["_centroid"]; cb = plane_b["_centroid"]
    dz1 = dz_away(plane_a, ca[0], ca[1], mid[0], mid[1], mid[2])
    dz2 = dz_away(plane_b, cb[0], cb[1], mid[0], mid[1], mid[2])

    if dz1 < 0 and dz2 < 0:
        kind = "narozie" if z_var >= 0.35 else "hreben"
    elif dz1 > 0 and dz2 > 0:
        kind = "uzlabie"
    else:
        kind = "narozie"  # nejednoznacne - bezpecny fallback (castejsi pripad)
    return kind, z_var


_TYPE_CODE = {"okap": "o", "stit": "s", "hreben": "h", "narozie": "n", "uzlabie": "u"}


def mesh_to_roof_planes_exact(
    mesh_path: str,
    distance_threshold: float = 0.06,
    min_inliers: int = 200,
    max_planes: int = 15,
    polygon_epsilon: float = 0.02,
    eave_snap_tol: float = 2.5,
    flat_roof_max_degree: float = 8.0,
) -> List[dict]:
    """
    Hlavny vstupny bod - nahrada za trimesh_roof_splitter.mesh_to_roof_planes.

    Args:
        mesh_path: cesta k .obj/.ply skenu strechy (realne metre)
        distance_threshold: RANSAC tolerancia (m) pri fitovani roviny
        min_inliers: min. pocet bodov aby sa rovina povazovala za platnu
        max_planes: max. pocet rovin, ktore sa bude hladat
        polygon_epsilon: relativny epsilon pre cv2.approxPolyDP zjednodusenie

    Returns:
        Zoznam planes v rovnakom formate ako trimesh_roof_splitter (id, type,
        area_m2, pitch_deg, vertices, edges[{id,type,length_m,start,end}]).
        Navyse kazdy plan ma 'low_confidence': bool (nizka hustota bodov -
        odporucana rucna kontrola) a kazda hrana 'exact': bool (True = presny
        priesecnik rovin, False = ponechana z povodnej hranice mraku - okap/stit).
    """
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    points = np.asarray(mesh.vertices)

    raw_planes = _multi_plane_ransac(
        points, distance_threshold=distance_threshold,
        min_inliers=min_inliers, max_planes=max_planes,
        flat_roof_max_degree=flat_roof_max_degree,
    )
    logger.info("exact_roof_planes: RANSAC nasiel %d rovin", len(raw_planes))

    # cisty hranicny polygon + centroid pre kazdu rovinu
    for p in raw_planes:
        pts3d = points[p["point_idx"]]
        p["_poly"] = _clean_boundary_polygon(pts3d, p["normal"], polygon_epsilon)
        p["_centroid"] = np.mean(p["_poly"], axis=0)
        # presny 'd' z cistej hranice (konzistentne s _poly)
        p["d"] = -p["normal"].dot(p["_centroid"])

    good_idx = [i for i, p in enumerate(raw_planes) if not p.get("_low_confidence")]
    low_conf_idx = [i for i in range(len(raw_planes)) if i not in good_idx]
    if low_conf_idx:
        logger.warning(
            "exact_roof_planes: %d rovin vyradenych pre nizku hustotu bodov "
            "(mozny artefakt) - indexy %s. Skontroluj rucne.",
            len(low_conf_idx), low_conf_idx,
        )

    adjacency = _confirmed_adjacent_planes(points, raw_planes)
    # obmedz susednost len na 'good' roviny
    adjacency = {k: v for k, v in adjacency.items() if k[0] in good_idx and k[1] in good_idx}
    logger.info("exact_roof_planes: overenych susednych dvojic rovin: %d", len(adjacency))

    neighbors = defaultdict(set)
    for (i, j) in adjacency:
        neighbors[i].add(j); neighbors[j].add(i)

    # --- orezanie kazdej roviny oproti overenym susedom ---
    # POISTKA (Claude fix v2): ak by orezanie jedneho suseda zmazalo vacsinu
    # zvysnej plochy naraz, je to podozrive - spravny orez pri skutocnom
    # susedovi zvycajne odreze len skromny kus pri okraji, nie vacsinu
    # polygonu. Toto zachytava pripady, ked side_val vyjde nespravne
    # (napr. centroid roviny lezi tesne pri nekonecnej rovine suseda, alebo
    # rovina ma malo bodov a jej fit je zasumeny) - vtedy sa sklopene
    # orezanie odreze NESPRAVNU (vacsiu) polovicu miesto maleho okraja,
    # a vysledny polygon sa "odtrhne" niekam inam (presne to, co bolo
    # vidno vo vieveri - R7/R1/R6 odletene od hlavneho zhluku).
    MIN_RETAINED_AREA_FRACTION = 0.20

    final_polys = {}
    for idx in good_idx:
        poly = raw_planes[idx]["_poly"]
        my_centroid = raw_planes[idx]["_centroid"]
        for nb in neighbors[idx]:
            q_normal, q_d = raw_planes[nb]["normal"], raw_planes[nb]["d"]
            q_centroid = raw_planes[nb]["_centroid"]
            # Fix: skus OBE strany orezu, vyber tu s vacsou plochou.
            # side_val zlyhava pre vnutorne roviny (R3) kde centroid je na opacnej strane.
            area_before = _polygon_area_3d(poly, raw_planes[idx]["normal"])
            clipped_pos = clip_polygon_by_plane(poly, q_centroid, q_normal, keep_positive=True)
            clipped_neg = clip_polygon_by_plane(poly, q_centroid, q_normal, keep_positive=False)
            area_pos = _polygon_area_3d(clipped_pos, raw_planes[idx]["normal"]) if len(clipped_pos) >= 3 else 0.0
            area_neg = _polygon_area_3d(clipped_neg, raw_planes[idx]["normal"]) if len(clipped_neg) >= 3 else 0.0

            if area_pos >= area_neg:
                clipped = clipped_pos
                area_after = area_pos
            else:
                clipped = clipped_neg
                area_after = area_neg

            # POISTKA: obe strany < MIN_RETAINED => preskoc (false adjacency)
            if area_before > 1e-6 and (area_after / area_before) < MIN_RETAINED_AREA_FRACTION:
                logger.warning(
                    "exact_roof_planes: orez P%d oproti P%d odstranil %.0f%% plochy "
                    "(obe strany < %.0f%%) - orez PRESKOCENY",
                    idx, nb, 100 * (1 - area_after / area_before),
                    100 * MIN_RETAINED_AREA_FRACTION,
                )
                clipped = poly  # preskoc tento orez, nechaj povodny poly

            poly = clipped if len(clipped) >= 3 else poly
        final_polys[idx] = poly

    # --- SNAP okapovych vrcholov (Claude fix v3) ---
    # Orezanie vyssie rieši len hrany, kde ma rovina OVERENEHO suseda
    # (hreben/narozie/uzlabie). Okapove vrcholy (kde strecha jednoducho
    # konci, ziadny sused na orez) ostavaju z povodneho NEZAVISLE fitovaneho
    # hranicneho polygonu kazdej roviny - a keďze kazda rovina sa fituje
    # samostatne z vlastneho mraku bodov, ten isty fyzicky roh budovy moze
    # vyjst v dvoch susednych rovinach o kus ineho posunuty (mm az niekolko
    # desiatok cm, niekedy viac). Vysledok: viditelne medzery ("hrany nie su
    # spojene"). Rieseние: sparuj a zlep (priemer) vrcholy z ROZNYCH rovin,
    # ktore su blizko seba - to iste co funguje manualne v Blenderi.
    EAVE_SNAP_TOL = 1.5  # metrov - max vzdialenost na sparovanie

    all_eave_verts = []  # (plane_idx, vert_idx, coord)
    for idx in good_idx:
        for vi, v in enumerate(final_polys[idx]):
            all_eave_verts.append([idx, vi, np.array(v, dtype=float)])

    n_ev = len(all_eave_verts)
    merged_flag = [False] * n_ev
    snap_count = 0
    for a in range(n_ev):
        if merged_flag[a]:
            continue
        pa, va, ca = all_eave_verts[a]
        cluster = [a]
        for b in range(a + 1, n_ev):
            if merged_flag[b]:
                continue
            pb, vb, cb = all_eave_verts[b]
            if pb == pa:
                continue  # len ROZNE roviny
            if np.linalg.norm(ca - cb) < EAVE_SNAP_TOL:
                cluster.append(b)
        if len(cluster) > 1:
            avg = np.mean([all_eave_verts[k][2] for k in cluster], axis=0)
            for k in cluster:
                pk, vk, _ = all_eave_verts[k]
                final_polys[pk][vk] = tuple(avg)
                merged_flag[k] = True
            snap_count += 1

    if snap_count:
        logger.info("exact_roof_planes: sparovanych/zlepenych %d okapovych vrcholov", snap_count)

    # --- klasifikacia hran ---
    # najprv oznac ktore hrany kazdej roviny su "exact" (vysledok orezania susedom)
    # tym, ze skontrolujeme, ci lezia blizko niektoreho zdielaneho priesecnika
    planes_out = []
    for out_i, idx in enumerate(good_idx):
        poly = final_polys[idx]
        n = len(poly)
        edges = []
        for ei in range(n):
            v1, v2 = poly[ei], poly[(ei + 1) % n]
            length_m = float(np.linalg.norm(np.array(v2) - np.array(v1)))
            if length_m < 0.3:
                continue
            mid = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2, (v1[2] + v2[2]) / 2)

            # over, ci je tato hrana totozna s niektorym overenym susedstvom
            edge_type = "okap"
            is_exact = False
            for nb in neighbors[idx]:
                key = (min(idx, nb), max(idx, nb))
                pi = adjacency.get(key)
                if pi is None:
                    continue
                # over, ci mid lezi blizko priamky priesecnika a v spravnom t-rozsahu
                rel = np.array(mid) - pi["point"]
                t = rel @ pi["direction"]
                perp = rel - t * pi["direction"]
                if np.linalg.norm(perp) < 0.30 or (np.linalg.norm(perp) < 0.60 and pi["t_lo"] - 0.5 <= t <= pi["t_hi"] + 0.5):
                    kind, z_var = _dihedral_type(pi, raw_planes[idx], raw_planes[nb])
                    edge_type = kind
                    is_exact = True
                    break

            code = _TYPE_CODE[edge_type]
            edges.append({
                "id": f"{code}{ei+1}",
                "type": code,
                "length_m": round(length_m, 3),
                "start": [round(v, 3) for v in v1],
                "end": [round(v, 3) for v in v2],
                "exact": is_exact,
            })

        normal = raw_planes[idx]["normal"]
        slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
        # plocha z cisteho (orezaneho) polygonu
        u, v = _plane_basis(normal)
        c = np.mean(poly, axis=0)
        local2d = np.array([[(np.array(p) - c) @ u, (np.array(p) - c) @ v] for p in poly])
        x, y = local2d[:, 0], local2d[:, 1]
        area = abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0

        planes_out.append({
            "id": f"R{out_i+1}",
            "type": ("plochá" if slope < flat_roof_max_degree else "sedlová" if 15 <= slope <= 45 else "valbová"),
            "area_m2": round(float(area), 2),
            "pitch_deg": round(slope, 1),
            "vertices": [[round(v, 3) for v in p] for p in poly],
            "edges": edges,
            "low_confidence": False,
        })

    for idx in low_conf_idx:
        pts3d = points[raw_planes[idx]["point_idx"]]
        normal = raw_planes[idx]["normal"]
        slope = float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1, 1))))
        planes_out.append({
            "id": f"R{len(planes_out)+1}_LOW_CONFIDENCE",
            "type": "neurcite - nizka hustota bodov, skontroluj rucne",
            "area_m2": round(_hull_area_approx(pts3d, normal), 2),
            "pitch_deg": round(slope, 1),
            "vertices": [],
            "edges": [],
            "low_confidence": True,
        })

    return planes_out


# ---------------------------------------------------------------------------
# Viewer HTML generator - kompatibilny s _triov_v9_viewer.html sablonou
# (rovnaka struktura, priamo start/end suradnice v edges, ziadne v1/v2 indexy)
# ---------------------------------------------------------------------------

_VIEWER_TEMPLATE_CACHE = None


def _load_viewer_template(template_path: Optional[str] = None) -> str:
    """
    Nacita HTML sablonu vieweru (hlada v9 viewer v output/ ak template_path
    nie je zadany explicitne). Sablona musi obsahovat presne riadok
    'const planes = [....];' ktory sa nahradi novymi datami.
    """
    global _VIEWER_TEMPLATE_CACHE
    if template_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", "..", "output", "_triov_v9_viewer.html"),
            os.path.join(here, "..", "visualization", "roof_3d_viewer.html"),
        ]
        for c in candidates:
            if os.path.exists(c):
                template_path = c
                break
    if template_path is None or not os.path.exists(template_path):
        raise FileNotFoundError(
            "Nenasla sa ziadna viewer sablona (_triov_v9_viewer.html). "
            "Zadaj template_path= explicitne, alebo skopiruj existujuci "
            "*_viewer.html do output/ priecinka."
        )
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def write_exact_viewer_html(
    planes: List[dict],
    address: str,
    output_path: str,
    template_path: Optional[str] = None,
) -> str:
    """
    Vygeneruje samostatny HTML viewer z vystupu mesh_to_roof_planes_exact().

    Na rozdiel od geometry_viewer.generate_geometry_viewer() (ktory ocakava
    edges s 'v1'/'v2' indexami do vertices_3d - NEZHODUJE sa s formatom co
    produkuje tento modul ani povodny trimesh_roof_splitter!), tato funkcia
    pouziva PRIAMO 'start'/'end' 3D suradnice v kazdej hrane, presne v tvare
    akom su v _triov_v9_viewer.html sablone.

    Args:
        planes: vystup z mesh_to_roof_planes_exact()
        address: adresa/nazov pre panel vo vieveri
        output_path: kam ulozit .html
        template_path: voliteľne, cesta k vlastnej sablone (inak sa hlada
            _triov_v9_viewer.html v output/)

    Returns:
        output_path
    """
    import re as _re

    template = _load_viewer_template(template_path)

    planes_for_js = [p for p in planes if not p.get("low_confidence")]
    planes_json = json.dumps(planes_for_js, ensure_ascii=False)

    new_html = _re.sub(
        r"const planes = \[.*?\];\n",
        f"const planes = {planes_json};\n",
        template,
        count=1,
        flags=_re.DOTALL,
    )
    if new_html == template:
        raise RuntimeError(
            "Nahradenie 'const planes = [...]' v sablone zlyhalo - "
            "sablona neobsahuje ocakavany format."
        )

    # nazov v <title> a paneli
    new_html = _re.sub(r"<title>.*?</title>", f"<title>Strecha - {address}</title>", new_html, count=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(new_html)

    n_low = len(planes) - len(planes_for_js)
    if n_low:
        logger.warning(
            "write_exact_viewer_html: %d rovin s low_confidence vynechanych "
            "z 3D zobrazenia (nemaju vertices/edges)", n_low
        )
    return output_path

