# -*- coding: utf-8 -*-
"""QA kontroly over geometriu a kontrakt. Nič nerieši — len hlási (fail-soft).

Kontroly vychádzajú z reálnych zlyhaní projektu:
  * `slope_flat` pri sklone 36,9° (vnútorný spor triedy a sklonu),
  * low_confidence rovina 410 m² (artefakt roztrúsenej roviny),
  * súčet plôch vs. pôdorys,
  * medzery medzi susednými plochami (nesmú existovať — požiadavka AGENTS.md),
  * každá hrana musí mať typ (o/h/n/u/s).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

FLAT_MAX_DEG = 8.0
SEDLO_MIN_DEG, SEDLO_MAX_DEG = 15.0, 45.0
AREA_SUM_TOLERANCE = 0.25     # 25 % odchýlka súčtu plôch od pôdorysu
GAP_TOL_M = 0.05              # považovať vrcholy za spoločné


def _plane_area(vertices: List[List[float]]) -> float:
    """Plocha polygónu v rovine X-Y (shoelace). Stačí na kontrolu poradia."""
    if len(vertices) < 3:
        return 0.0
    s = 0.0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i][0], vertices[i][1]
        x2, y2 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def check_class_pitch(model) -> List[Dict[str, Any]]:
    """Trieda roviny nesmie odporovať sklonu (dôkaz: slope_flat pri 36,9°)."""
    issues: List[Dict[str, Any]] = []
    for p in model.planes:
        if p.low_confidence:
            continue
        if p.type == "plochá" and p.pitch_deg > FLAT_MAX_DEG:
            issues.append({"check": "class_pitch", "plane": p.id, "severity": "error",
                           "detail": f"type=plochá ale pitch_deg={p.pitch_deg}"})
        if p.type == "sedlová" and not (SEDLO_MIN_DEG <= p.pitch_deg <= SEDLO_MAX_DEG):
            issues.append({"check": "class_pitch", "plane": p.id, "severity": "warning",
                           "detail": f"type=sedlová ale pitch_deg={p.pitch_deg} mimo 15–45°"})
    return issues


def check_area_sum(model) -> List[Dict[str, Any]]:
    """Súčet plôch rovín vs. plocha strechy v kontrakte (ak existuje)."""
    issues: List[Dict[str, Any]] = []
    if model.roof_area_m2 is None:
        return issues
    total = sum(p.area_m2 for p in model.planes if not p.low_confidence)
    if total <= 0:
        return issues
    rel = abs(total - model.roof_area_m2) / max(total, model.roof_area_m2)
    if rel > AREA_SUM_TOLERANCE:
        issues.append({"check": "area_sum", "severity": "error",
                       "detail": f"súčet rovín={total:.1f} m² vs area={model.roof_area_m2:.1f} m² (rel={rel:.2f})"})
    return issues


def check_low_confidence(model) -> List[Dict[str, Any]]:
    """Roztrúsené roviny musia byť flagované (dôkaz: low_conf 410 m² artefakt)."""
    issues: List[Dict[str, Any]] = []
    for p in model.planes:
        if p.low_confidence and p.area_m2 > 50:
            issues.append({"check": "low_confidence", "plane": p.id, "severity": "warning",
                           "detail": f"low_conf rovina má {p.area_m2:.1f} m² — skontroluj manuálne"})
        if p.low_confidence and p.edges:
            issues.append({"check": "low_confidence", "plane": p.id, "severity": "error",
                           "detail": "low_conf rovina nesie hrany (mala by byť prázdna)"})
    return issues


def check_gaps(model, gap_tol_m: float = GAP_TOL_M) -> List[Dict[str, Any]]:
    """Susedné polygóny nesmú mať medzery: hľadám vrcholy bližšie než gap_tol_m,
    ktoré nie sú totožné (presný snap medzi rovinami)."""
    issues: List[Dict[str, Any]] = []
    pts: List[tuple] = []
    for p in model.planes:
        for v in p.vertices:
            pts.append((p.id, float(v[0]), float(v[1])))
    n = len(pts)
    if n > 4000:  # ochrana proti O(n²) na veľkých modeloch
        return issues
    checked = set()
    for i in range(n):
        pi, xi, yi = pts[i]
        for j in range(i + 1, n):
            pj, xj, yj = pts[j]
            if pi == pj:
                continue
            d = math.hypot(xi - xj, yi - yj)
            if gap_tol_m < d <= 0.30 and (pi, pj) not in checked:
                checked.add((pi, pj))
                issues.append({"check": "gaps", "severity": "warning",
                               "detail": f"{pi}↔{pj}: vrcholy {d:.3f} m od seba (nesúsnutné hrany)"})
    return issues


def check_edge_types(model) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    allowed = {"o", "h", "n", "u", "s"}
    counts: Dict[str, int] = {}
    for p in model.planes:
        for e in p.edges:
            counts[e.type] = counts.get(e.type, 0) + 1
            if e.type not in allowed:
                issues.append({"check": "edge_types", "plane": p.id, "edge": e.id,
                               "severity": "error", "detail": f"neznámy typ hrany {e.type!r}"})
    if model.planes and not counts:
        issues.append({"check": "edge_types", "severity": "warning",
                       "detail": "model nemá žiadne klasifikované hrany"})
    return issues


def check_area_vs_footprint(model, max_ratio: float = 1.6) -> List[Dict[str, Any]]:
    """Súčet plôch rovín nesmie výrazne prevýšiť pôdorys — inak sa do modelu
    dostali susedné budovy (presne to sa stalo: 850 m² rovín vs 344 m² pôdorysu)."""
    issues: List[Dict[str, Any]] = []
    if not model.footprint_m2 or not model.planes:
        return issues
    total = sum(p.area_m2 for p in model.planes if not p.low_confidence)
    ratio = total / model.footprint_m2
    if ratio > max_ratio:
        issues.append({"check": "area_vs_footprint", "severity": "error",
                       "detail": f"roviny {total:.0f} m² = {ratio:.2f}× pôdorys {model.footprint_m2:.0f} m² "
                                 f"(pravdepodobne viac budov v jednom modeli)"})
    elif ratio < 0.5:
        issues.append({"check": "area_vs_footprint", "severity": "warning",
                       "detail": f"roviny {total:.0f} m² je len {ratio:.2f}× pôdorysu — chýbajú plochy?"})
    return issues


def run_all_checks(model) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    issues += check_class_pitch(model)
    issues += check_eave_horizontal(model)
    issues += check_gable_on_shared_edge(model)
    issues += check_duplicate_edges(model)
    issues += check_edge_type_consistency(model)
    issues += check_plane_has_eave(model)
    issues += check_areas_true(model)
    issues += check_area_sum(model)
    issues += check_area_vs_footprint(model)
    issues += check_low_confidence(model)
    issues += check_gaps(model)
    issues += check_edge_types(model)
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    return {
        "errors": errors,
        "warnings": warnings,
        "counts": {"errors": len(errors), "warnings": len(warnings)},
        "verdict": "FAIL" if errors else ("WARN" if warnings else "PASS"),
    }

# ─── kontroly konzistencie hrán (audit 2026-10-05) ───────────────────────────

def _seg_mid(e):
    return ((e.start[0] + e.end[0]) / 2.0, (e.start[1] + e.end[1]) / 2.0)


def _seg_dir(e):
    dx, dy = e.end[0] - e.start[0], e.end[1] - e.start[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def _same_line(e1, e2, dist_tol: float = 0.30, ang_tol_deg: float = 8.0,
               min_overlap_m: float = 1.0) -> bool:
    """Ležia dve hrany na tej istej čiare (a prekrývajú sa)?"""
    d1x, d1y = _seg_dir(e1)
    d2x, d2y = _seg_dir(e2)
    ang = math.degrees(math.acos(min(1.0, abs(d1x * d2x + d1y * d2y))))
    if ang > ang_tol_deg:
        return False
    mx, my = _seg_mid(e2)
    # kolmá vzdialenosť stredu e2 od línie e1
    dist = abs(-d1y * (mx - e1.start[0]) + d1x * (my - e1.start[1]))
    if dist > dist_tol:
        return False
    # prekrytie pozdĺž línie e1
    ts = []
    for px, py in ((e2.start[0], e2.start[1]), (e2.end[0], e2.end[1])):
        ts.append(d1x * (px - e1.start[0]) + d1y * (py - e1.start[1]))
    L1 = math.hypot(e1.end[0] - e1.start[0], e1.end[1] - e1.start[1])
    lo, hi = min(ts), max(ts)
    overlap = max(0.0, min(hi, L1) - max(lo, 0.0))
    return overlap >= min_overlap_m


def check_eave_horizontal(model, max_deg: float = 6.0) -> List[Dict[str, Any]]:
    """Odvk ap musí byť takmer vodorovný (vedie po vrstevnici)."""
    issues = []
    for p in model.planes:
        for e in p.edges:
            if e.type != "o":
                continue
            L2 = math.hypot(e.end[0] - e.start[0], e.end[1] - e.start[1])
            dz = abs(float(e.end[2]) - float(e.start[2]))
            if L2 < 0.5:
                continue
            slope = math.degrees(math.atan2(dz, L2))
            if slope > max_deg:
                issues.append({"check": "eave_horizontal", "plane": p.id, "edge": e.id,
                               "severity": "error",
                               "detail": f"odkvap stúpa {slope:.1f}° ({dz:.2f} m / {L2:.1f} m) — skôr štít"})
    return issues


def check_gable_on_shared_edge(model) -> List[Dict[str, Any]]:
    """'s' (štít) je voľná hrana — nesmie ležať na hrane inej roviny."""
    issues = []
    for i, p in enumerate(model.planes):
        for e in p.edges:
            if e.type != "s":
                continue
            for j, q in enumerate(model.planes):
                if i == j:
                    continue
                for f in q.edges:
                    if _same_line(e, f):
                        issues.append({"check": "gable_shared", "plane": p.id, "edge": e.id,
                                       "severity": "error",
                                       "detail": f"štít {e.id} leží na hrane {f.id} roviny {q.id} "
                                                 f"(typ {f.type}) — spoločná hrana nemôže byť štít"})
    return issues


def check_duplicate_edges(model, min_overlap_m: float = 1.0) -> List[Dict[str, Any]]:
    """Tá istá fyzická hrana 2× v kontrakte (duplicitný zápis)."""
    issues = []
    seen = []
    for p in model.planes:
        for e in p.edges:
            for (pid, eid, e0) in seen:
                if pid == p.id and _same_line(e, e0):
                    issues.append({"check": "duplicate_edges", "plane": p.id, "edge": e.id,
                                   "severity": "warning",
                                   "detail": f"hrana {e.id} je na tej istej čiare ako {eid} "
                                             f"v tej istej rovine (duplicita)"})
                    break
            seen.append((p.id, e.id, e))
    return issues


def check_edge_type_consistency(model) -> List[Dict[str, Any]]:
    """Tá istá hrana (čiara) musí mať rovnaký typ v oboch susedných rovinách."""
    issues = []
    for i, p in enumerate(model.planes):
        for e in p.edges:
            if e.type == "s":
                continue
            for j, q in enumerate(model.planes):
                if i >= j:
                    continue
                for f in q.edges:
                    if f.type == "s" or e.type == f.type:
                        continue
                    if _same_line(e, f):
                        issues.append({"check": "edge_type_consistency", "plane": p.id, "edge": e.id,
                                       "severity": "error",
                                       "detail": f"{e.id} je '{e.type}' v {p.id}, ale {f.id} je "
                                                 f"'{f.type}' v {q.id} — tá istá hrana"})
    return issues


def check_plane_has_eave(model, min_pitch_deg: float = 8.0) -> List[Dict[str, Any]]:
    """Šikmá rovina musí mať odkvap (o)."""
    issues = []
    for p in model.planes:
        if p.low_confidence or p.pitch_deg < min_pitch_deg:
            continue
        if not any(e.type == "o" for e in p.edges):
            issues.append({"check": "plane_has_eave", "plane": p.id, "severity": "warning",
                           "detail": f"šikmá rovina ({p.pitch_deg:.1f}°) nemá odkvapovú hranu"})
    return issues


def check_areas_true(model, tol_rel: float = 0.05) -> List[Dict[str, Any]]:
    """Upozorni, že 'area_m2' je pôdorysný priemet, nie plocha strechy.

    Skutočná plocha = pôdorys / cos(sklon). Pre kalkuláciu materiálu je to podstatný rozdiel
    (pri 26° je to +11 %).
    """
    issues = []
    total_plan = sum(p.area_m2 for p in model.planes if not p.low_confidence)
    total_true = 0.0
    for p in model.planes:
        if p.low_confidence:
            continue
        c = math.cos(math.radians(min(89.0, float(p.pitch_deg or 0.0))))
        total_true += p.area_m2 / c if c > 0.1 else p.area_m2
    if total_plan > 0:
        rel = (total_true - total_plan) / total_plan
        if rel > tol_rel:
            issues.append({"check": "areas_true", "severity": "warning",
                           "detail": f"skutočná plocha strechy (šikmá) je {total_true:.1f} m², "
                                     f"pôdorysný súčet {total_plan:.1f} m² (+{100*rel:.0f} %) — "
                                     f"použiť pre kalkuláciu materiálu"})
    return issues
