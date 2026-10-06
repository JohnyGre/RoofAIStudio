# -*- coding: utf-8 -*-
"""QA kontroly over geometriu a kontrakt. Nič nerieši — len hlási (fail-soft).

Kontroly vychádzajú z reálnych zlyhaní projektu:
  * `slope_flat` pri sklone 36,9° (vnútorný spor triedy a sklonu),
  * low_confidence rovina 410 m² (artefakt roztrúsenej roviny),
  * súčet plôch vs. pôdorys,
  * medzery medzi susednými plochami (nesmú existovať — požiadavka AGENTS.md),
  * každá hrana musí mať typ (o/h/n/u/s),
  * konzistencia hrán medzi rovinami (audit Átriová 9309/16: štít na cudzej hrane,
    duplicitné hrany, stúpajúci „odkvap", pôdorysná plocha vydávaná za plochu strechy).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

FLAT_MAX_DEG = 8.0
SEDLO_MIN_DEG, SEDLO_MAX_DEG = 15.0, 45.0
AREA_SUM_TOLERANCE = 0.25     # 25 % odchýlka súčtu plôch od pôdorysu
GAP_TOL_M = 0.05              # považovať vrcholy za spoločné

# --- konzistencia hrán ---
EAVE_MAX_DEG = 8.0            # odkvap musí byť takmer vodorovný. Dáta (Triova, Beluj): skutočné odkvapy
                              # majú sklon 0–6,6° (kvôli posunu rohu polygónu), štíty a bočnice ≥ 11°.
DEGENERATE_AREA_M2 = 1.0      # rovina menšia než 1 m² je artefakt (rovnaký prah ako legacy low_confidence)
COINCIDE_TOL_M = 0.6          # dve hrany „ležia na tej istej čiare" (XY)
COINCIDE_MAX_ANGLE_DEG = 15.0
COINCIDE_MIN_OVERLAP = 0.5    # prekrytie ≥ 50 % kratšej hrany (kolineárne susedné úseky sa nerátajú)
PITCHED_MIN_DEG = FLAT_MAX_DEG
AREA_TRUE_INFO_MIN_REL = 0.05  # info o skutočnej ploche len ak sa líši > 5 % (sklon > ~18°)

# Závažnosť na jednom mieste. Štrukturálne rozpory v klasifikácii hrán sú chyby (CI ich
# musí zastaviť), duplicity a chýbajúci odkvap varovania, plocha len informácia (nemení verdikt).
SEVERITY = {
    "eave_horizontal": "error",
    "gable_shared": "error",
    "edge_type_consistency": "error",
    "duplicate_edges": "warning",
    "plane_has_eave": "warning",
    "areas_true": "info",
}


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


# ---------------------------------------------------------------------------
# Konzistencia hrán
# ---------------------------------------------------------------------------

def _seg_xy(e):
    return (float(e.start[0]), float(e.start[1])), (float(e.end[0]), float(e.end[1]))


def _coincide(e1, e2, tol: float = COINCIDE_TOL_M) -> bool:
    """Ležia dve hrany na tej istej čiare v pôdoryse? (rovnobežné, blízko, prekrývajú sa)

    Kolineárne susedné úseky (napr. dva kusy jedného hrebeňa) sa NErátajú — chýba im prekrytie.
    """
    a0, a1 = _seg_xy(e1)
    b0, b1 = _seg_xy(e2)
    la = math.hypot(a1[0] - a0[0], a1[1] - a0[1])
    lb = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
    if la < 1e-6 or lb < 1e-6:
        return False
    u = ((a1[0] - a0[0]) / la, (a1[1] - a0[1]) / la)
    v = ((b1[0] - b0[0]) / lb, (b1[1] - b0[1]) / lb)
    if abs(u[0] * v[0] + u[1] * v[1]) < math.cos(math.radians(COINCIDE_MAX_ANGLE_DEG)):
        return False

    def perp(p, o, d):
        return abs((p[0] - o[0]) * (-d[1]) + (p[1] - o[1]) * d[0])

    dist = max(perp(b0, a0, u), perp(b1, a0, u), perp(a0, b0, v), perp(a1, b0, v))
    if dist > tol:
        return False
    t0 = (b0[0] - a0[0]) * u[0] + (b0[1] - a0[1]) * u[1]
    t1 = (b1[0] - a0[0]) * u[0] + (b1[1] - a0[1]) * u[1]
    overlap = min(la, max(t0, t1)) - max(0.0, min(t0, t1))
    return overlap >= COINCIDE_MIN_OVERLAP * min(la, lb)


def _plane_z_fn(vertices):
    """Výška roviny v (x, y) z jej vrcholov; None ak sa nedá (stdlib fallback = ulož. z)."""
    try:
        import numpy as np
    except ImportError:
        return None
    pts = [v for v in vertices if len(v) >= 3]
    if len(pts) < 3:
        return None
    P = np.asarray(pts, dtype=float)
    c = P[:, :2].mean(axis=0)          # centrovanie: S-JTSK súradnice sú ~1e6
    A = np.c_[P[:, 0] - c[0], P[:, 1] - c[1], np.ones(len(P))]
    try:
        coef, _, rank, _ = np.linalg.lstsq(A, P[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or float(np.abs(A @ coef - P[:, 2]).max()) > 0.25:
        return None                    # polygón nie je rovinný — neveríme mu
    return lambda x, y: float(coef[0] * (x - c[0]) + coef[1] * (y - c[1]) + coef[2])


def _edge_slope_deg(edge) -> float:
    """Sklon hrany podľa z-hodnôt uložených na jej koncoch."""
    (x0, y0), (x1, y1) = _seg_xy(edge)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 0.1:
        return 0.0
    return math.degrees(math.atan2(abs(float(edge.end[2]) - float(edge.start[2])), length))


def _eave_slope_deg(plane, edge) -> float:
    """Sklon hrany ako väčší z dvoch odhadov: z uložených výšok hrany a z výšok ROVINY
    v koncových bodoch. Uložený z sám nestačí: pôvodný chybný odkvap mal oba konce v rovnakej
    výške, hoci hrana v rovine stúpala. Naopak z roviny samotnej nechytí hranu s nesprávnym z."""
    (x0, y0), (x1, y1) = _seg_xy(edge)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 0.1:
        return 0.0
    slope = _edge_slope_deg(edge)
    fn = _plane_z_fn(plane.vertices)
    if fn is not None:
        slope = max(slope, math.degrees(math.atan2(abs(fn(x1, y1) - fn(x0, y0)), length)))
    return slope


def check_eave_horizontal(model, max_deg: float = EAVE_MAX_DEG) -> List[Dict[str, Any]]:
    """Odkvap (o) musí byť takmer vodorovný — stúpajúca hrana je štít, nie odkvap."""
    issues: List[Dict[str, Any]] = []
    for p in model.planes:
        for e in p.edges:
            if e.type != "o":
                continue
            ang = _eave_slope_deg(p, e)
            if ang > max_deg:
                issues.append({"check": "eave_horizontal", "plane": p.id, "edge": e.id,
                               "severity": SEVERITY["eave_horizontal"],
                               "detail": f"odkvap {e.id} ({e.length_m:.1f} m) stúpa pod {ang:.1f}° "
                                         f"(> {max_deg:.0f}°) — je to štít?"})
    return issues


def check_gable_on_shared_edge(model) -> List[Dict[str, Any]]:
    """Štít (s) je voľná hrana — nesmie ležať na hrane inej roviny."""
    issues: List[Dict[str, Any]] = []
    for pa in model.planes:
        for ea in pa.edges:
            if ea.type != "s":
                continue
            for pb in model.planes:
                if pb.id == pa.id:
                    continue
                other = next((eb for eb in pb.edges if _coincide(ea, eb)), None)
                if other is not None:
                    issues.append({"check": "gable_shared", "plane": pa.id, "edge": ea.id,
                                   "severity": SEVERITY["gable_shared"],
                                   "detail": f"štít {pa.id}.{ea.id} leží na hrane {pb.id}.{other.id} "
                                             f"(typ {other.type}) — spoločná hrana nemôže byť štít"})
                    break          # jedno hlásenie na štít, nie na každú susednú rovinu
    return issues


check_gable_shared = check_gable_on_shared_edge        # alias (pôvodný názov v mojej verzii)


def check_duplicate_edges(model) -> List[Dict[str, Any]]:
    """Tá istá hrana dvakrát v jednej rovine (polygónová + presná z priesečníc)."""
    issues: List[Dict[str, Any]] = []
    for p in model.planes:
        es = p.edges
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                if _coincide(es[i], es[j]):
                    issues.append({"check": "duplicate_edges", "plane": p.id, "edge": es[j].id,
                                   "severity": SEVERITY["duplicate_edges"],
                                   "detail": f"{p.id}: hrany {es[i].id} ({es[i].type}) a "
                                             f"{es[j].id} ({es[j].type}) ležia na tej istej čiare"})
    return issues


def check_edge_type_consistency(model) -> List[Dict[str, Any]]:
    """Tá istá spoločná hrana musí mať v oboch susedných rovinách rovnaký typ.
    (Páry so štítom rieši gable_shared, aby sa nehlásilo dvakrát.)"""
    issues: List[Dict[str, Any]] = []
    planes = model.planes
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            for ea in planes[i].edges:
                for eb in planes[j].edges:
                    if ea.type == eb.type or "s" in (ea.type, eb.type):
                        continue
                    if _coincide(ea, eb):
                        issues.append({"check": "edge_type_consistency", "plane": planes[i].id,
                                       "edge": ea.id, "severity": SEVERITY["edge_type_consistency"],
                                       "detail": f"{planes[i].id}.{ea.id}={ea.type!r} vs "
                                                 f"{planes[j].id}.{eb.id}={eb.type!r} — tá istá hrana, "
                                                 f"rôzny typ"})
    return issues


def check_plane_has_eave(model, min_pitch_deg: float = PITCHED_MIN_DEG) -> List[Dict[str, Any]]:
    """Šikmá rovina má spravidla vodorovný odkvap. Preskakujú sa roviny bez akýchkoľvek hrán
    (legacy meta) a degenerované roviny (< 1 m²). Malé strmé roviny (vikier, < 15 m²) často
    odvodňujú do susedov — pre ne je to len informácia."""
    issues: List[Dict[str, Any]] = []
    for p in model.planes:
        if (p.low_confidence or p.pitch_deg < min_pitch_deg or not p.edges
                or p.area_m2 < DEGENERATE_AREA_M2):
            continue
        if any(e.type == "o" and _eave_slope_deg(p, e) <= EAVE_MAX_DEG for e in p.edges):
            continue
        small = p.type == "vikier"
        issues.append({"check": "plane_has_eave", "plane": p.id,
                       "severity": "info" if small else SEVERITY["plane_has_eave"],
                       "detail": f"šikmá rovina ({p.pitch_deg:.1f}°) nemá vodorovný odkvap"
                                 + (" (malá strmá rovina — môže odvodňovať do susedov)" if small else "")})
    return issues


def check_areas_true(model, tol_rel: float = AREA_TRUE_INFO_MIN_REL) -> List[Dict[str, Any]]:
    """`area_m2` je pôdorysný priemet (shoelace v XY). Skutočná plocha = priemet / cos(sklon).
    Ak kontrakt už nesie `area_true_m2`, upozornenie netreba — kalkulácia ho vidí priamo."""
    issues: List[Dict[str, Any]] = []
    planes = [p for p in model.planes if not p.low_confidence and p.area_m2 > 0]
    if not planes or all(getattr(p, "area_true_m2", None) for p in planes):
        return issues
    planar = sum(p.area_m2 for p in planes)
    true = sum(p.area_m2 / max(math.cos(math.radians(min(89.0, p.pitch_deg))), 0.2) for p in planes)
    rel = true / planar - 1.0
    declared = model.roof_area_m2
    if rel > tol_rel and (declared is None or abs(planar - declared) / planar < 0.02):
        issues.append({"check": "areas_true", "severity": SEVERITY["areas_true"],
                       "detail": f"area_m2 je pôdorysný priemet ({planar:.1f} m²); skutočná plocha "
                                 f"strechy ≈ {true:.1f} m² (+{rel * 100:.0f} %) — pre materiál použi tú"})
    return issues


def run_all_checks(model) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    issues += check_class_pitch(model)
    issues += check_area_sum(model)
    issues += check_area_vs_footprint(model)
    issues += check_low_confidence(model)
    issues += check_gaps(model)
    issues += check_edge_types(model)
    issues += check_eave_horizontal(model)
    issues += check_gable_on_shared_edge(model)
    issues += check_duplicate_edges(model)
    issues += check_edge_type_consistency(model)
    issues += check_plane_has_eave(model)
    issues += check_areas_true(model)
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]
    info = [i for i in issues if i.get("severity") == "info"]     # nemení verdikt
    return {
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "counts": {"errors": len(errors), "warnings": len(warnings), "info": len(info)},
        "verdict": "FAIL" if errors else ("WARN" if warnings else "PASS"),
    }
