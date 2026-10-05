# -*- coding: utf-8 -*-
"""Most medzi desktopovou aplikáciou a kontraktom v3: klasifikované hrany.

Prečo: `roofai_desktop_v2.py` si roviny rieši vlastným RANSAC-om a do kontraktu
posielal len legacy `*_meta.json` (sklon, azimut, plocha) — bez hrán. QA preto
vždy hlásila WARN „model nemá žiadne klasifikované hrany". CLI `tools/run_v3.py`
hrany do kontraktu zapisuje, desktop nie.

Tento modul NEPRIDÁVA novú klasifikáciu. Používa ten istý, už overený
`engine.intersection_edges` (hrebeň / nárožie / úžľabie z priesečníc rovín
a reálnych bodov mračna) a výsledok zapíše do kontraktu rovnakou konvenciou ako
`run_v3.py` (hrana sa uloží k obom susedným rovinám).

Zatiaľ sa dopĺňajú len presné hrany `h` / `n` / `u`. Odkvap (`o`) a štít (`s`)
vyžadujú polygóny rovín, ktoré legacy meta nenesie (ďalší krok).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.core import contract, engine


def desktop_planes_to_engine(planes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Desktopový formát rovín {n, coef, pts, ...} → formát enginu {normal, d, points}.

    Rovina je n·p + d = 0, kde n má kladné z (rovnaká konvencia ako v engine.py:
    spád ide v smere (n_x, n_y)). `d` počítame z ťažiska bodov, nie z `coef`,
    aby výsledok nezávisel od mierky koeficientov.
    """
    out: List[Dict[str, Any]] = []
    for pl in planes:
        pts = np.asarray(pl["pts"], dtype=float)
        n = np.asarray(pl["n"], dtype=float)
        n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        d = -float(n @ pts.mean(axis=0))
        out.append({"normal": n, "d": d, "points": pts})
    return out


def exact_edges(planes_engine: List[Dict[str, Any]], **kw) -> List[Dict[str, Any]]:
    """Presné hrany z priesečníc rovín (typy h / n / u). Tenký wrapper nad enginom."""
    if len(planes_engine) < 2:
        return []
    return engine.intersection_edges(planes_engine, **kw)


def attach_edges(model: "contract.RoofModel",
                 plane_src: Sequence[Optional[int]],
                 edges: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Doplní hrany do rovín kontraktu (in-place).

    plane_src[i] = index zdrojovej (RANSAC) roviny pre rovinu `model.planes[i]`.
    Jedna zdrojová rovina môže mať v kontrakte viac záznamov (viac kontúr) —
    hrany dostane len ten s najväčšou plochou, aby sa nezdvojili.
    Hrana sa zapíše k obom susedným rovinám pod rovnakým id (je to tá istá hrana).
    """
    target: Dict[int, int] = {}
    for i, p in enumerate(model.planes):
        s = plane_src[i] if i < len(plane_src) else None
        if s is None or p.low_confidence:
            continue
        if s not in target or p.area_m2 > model.planes[target[s]].area_m2:
            target[s] = i

    attached = 0
    for k, e in enumerate(edges, 1):
        rec_id = f"X{e['type']}{k}"
        hit = False
        for s in e.get("pair", []):
            if s in target:
                model.planes[target[s]].edges.append(contract.EdgeRecord(
                    id=rec_id, type=e["type"], length_m=round(float(e["length_m"]), 3),
                    start=[float(v) for v in e["start"]], end=[float(v) for v in e["end"]],
                    exact=bool(e.get("exact", True))))
                hit = True
        attached += int(hit)

    by_type = Counter(e["type"] for e in edges)
    return {"edges_found": len(edges), "edges_attached": attached, "by_type": dict(by_type)}


def add_edges_from_desktop(model: "contract.RoofModel",
                           plane_areas: Sequence[Dict[str, Any]],
                           planes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Jedno volanie pre desktop: RANSAC roviny + `roviny` z meta → hrany v kontrakte.

    `plane_areas[i]["src"]` je index zdrojovej roviny (zapisuje `topdown_mesh`).
    Ak chýba (starý meta), hrany sa nedoplnia a funkcia to nahlási, nehádže.
    """
    plane_src = [pa.get("src") for pa in plane_areas]
    if not any(s is not None for s in plane_src):
        return {"edges_found": 0, "edges_attached": 0, "by_type": {},
                "note": "plane_areas bez kľúča 'src' — hrany sa nedoplnili"}
    eng_planes = desktop_planes_to_engine(planes)
    edges = exact_edges(eng_planes)
    return attach_edges(model, plane_src, edges)
