# -*- coding: utf-8 -*-
"""Odkvapové hrany (o) nad hotovým kontraktom — úloha 1 z handoveru.

Diagnóza reálnych kontraktov (Triova, Beluj) ukázala, že chýbajúci odkvap mal 3 príčiny:

  A) Skutočný odkvap JE medzi stranami polygónu, ale je typu `s`, lebo má sklon 6,1–6,6°
     (roh polygónu je posunutý vzhľadom na vrstevnicu) a prah bol 6,0°.  → povýšiť s → o.
  B) Odkvap bol vyhodený deduplikáciou, lebo na jeho čiare vznikla „presná" hrana z priesečnice
     s degenerovanou rovinou (Beluj R5: 0,21 m², Triova R1: 0 m²) — vodorovné „úžľabie".
     → degenerované roviny najprv vyradiť (low_confidence, bez hrán), potom odkvap doplniť.
  C) Malé strmé roviny (vikier) odkvap mať nemusia → QA to hlási len ako info.

Kandidát na odkvap = strana polygónu roviny, ktorá
  1. je vodorovná podľa výšok ROVINY v oboch koncoch (≤ qa.EAVE_MAX_DEG),
  2. leží pri najnižšom bode roviny (nižší koniec najviac LOW_BAND_M nad ním),
  3. nie je spoločná s hranou/polygónom inej (nedegenerovanej) roviny,
  4. nie je už zapísaná ako `o` ani ako vnútorná hrana (h/n/u) tej istej roviny.
Ak na nej leží `s`, povýši sa na `o`; inak sa pridá nová `o` s výškami z roviny.

Obmedzenie: odkvap, ktorý polygón stratil úplne (nie je medzi jeho stranami), sa tu nenájde.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from app.core import contract, qa

LOW_BAND_M = 0.5        # nižší koniec odkvapu leží najviac 0,5 m nad najnižším bodom roviny
MIN_EAVE_LEN_M = 1.0    # kratšie úseky nie sú odkvap (šum po orezaní)
_INNER = ("h", "n", "u")


def _poly_edges(vertices):
    n = len(vertices)
    for i in range(n):
        yield vertices[i], vertices[(i + 1) % n]


def _seg(a, b, za, zb, eid="tmp", t="o", exact=False):
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return contract.EdgeRecord(id=eid, type=t, length_m=round(length, 3),
                               start=[float(a[0]), float(a[1]), float(za)],
                               end=[float(b[0]), float(b[1]), float(zb)], exact=exact)


def _is_degenerate(p) -> bool:
    return p.low_confidence or p.area_m2 < qa.DEGENERATE_AREA_M2


def quarantine_degenerate_planes(model, min_area_m2: float = qa.DEGENERATE_AREA_M2) -> List[str]:
    """Roviny < 1 m² sú artefakty: označí ich `low_confidence`, zmaže im hrany a odstráni
    susedom presné hrany, ktoré vznikli len priesečníkom s nimi (inak vzniká vodorovné „úžľabie"
    na čiare skutočného odkvapu). Id rovín sa nemenia."""
    dropped_exact = []
    quarantined: List[str] = []
    for p in model.planes:
        if p.low_confidence or p.area_m2 >= min_area_m2:
            continue
        dropped_exact += [e for e in p.edges if e.exact]
        p.edges = []
        p.low_confidence = True
        quarantined.append(p.id)
    if quarantined:
        for q in model.planes:
            if q.id in quarantined:
                continue
            q.edges = [e for e in q.edges
                       if not (e.exact and any(qa._coincide(e, d) for d in dropped_exact))]
    return quarantined


def _other_planes_segments(model, plane_id):
    """Hrany a polygónové strany ostatných (nedegenerovaných) rovín — test „spoločná hrana"."""
    segs = []
    for q in model.planes:
        if q.id == plane_id or _is_degenerate(q):
            continue
        segs.extend(q.edges)
        for a, b in _poly_edges(q.vertices):
            segs.append(_seg(a, b, a[2], b[2], t="?"))
    return segs


def _free_id(prefix: str, used: set) -> str:
    n = 1
    while f"{prefix}{n}" in used:
        n += 1
    return f"{prefix}{n}"


def add_missing_eaves(model, max_deg: float = qa.EAVE_MAX_DEG,
                      low_band_m: float = LOW_BAND_M) -> List[Dict[str, Any]]:
    """Rovinám bez vodorovného odkvapu doplní/povýši `o` hranu (in-place).
    Vracia zoznam {plane, edge, length_m, action: 'added'|'promoted'}."""
    done: List[Dict[str, Any]] = []
    for p in model.planes:
        if _is_degenerate(p) or p.pitch_deg < qa.PITCHED_MIN_DEG or len(p.vertices) < 3:
            continue
        if any(e.type == "o" and qa._eave_slope_deg(p, e) <= max_deg for e in p.edges):
            continue
        fn = qa._plane_z_fn(p.vertices)
        if fn is None:
            continue
        zmin = min(float(v[2]) for v in p.vertices if len(v) >= 3)
        others = _other_planes_segments(model, p.id)
        used = {e.id for e in p.edges}
        for a, b in _poly_edges(p.vertices):
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            if length < MIN_EAVE_LEN_M:
                continue
            za, zb = fn(a[0], a[1]), fn(b[0], b[1])
            if math.degrees(math.atan2(abs(zb - za), length)) > max_deg:
                continue
            if min(za, zb) - zmin > low_band_m:
                continue
            cand = _seg(a, b, za, zb)
            own = [e for e in p.edges if qa._coincide(cand, e)]
            if any(e.type == "o" or e.type in _INNER for e in own):
                continue
            if any(qa._coincide(cand, o) for o in others):
                continue
            gable = next((e for e in own if e.type == "s"), None)
            if gable is not None:                      # A) štít s malým sklonom → odkvap
                used.discard(gable.id)
                gable.type = "o"
                gable.id = _free_id("o", used)
                used.add(gable.id)
                done.append({"plane": p.id, "edge": gable.id, "length_m": gable.length_m, "action": "promoted"})
            else:                                      # B) odkvap chýba úplne → pridať
                cand.id = _free_id("o", used)
                used.add(cand.id)
                p.edges.append(cand)
                done.append({"plane": p.id, "edge": cand.id, "length_m": cand.length_m, "action": "added"})
    return done
