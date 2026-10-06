# -*- coding: utf-8 -*-
"""Finálna očista hrán v kontrakte (nahrádza inline blok „4c" v run_v3.py).

Dva kroky, obidva deterministické a fail-soft:

1. DUPLICITA S VLASTNOU PRESNOU HRANOU (X): polygónová hrana, na ktorej leží presná hrana tej
   istej roviny, sa neruší celá, ale ORIEZNE o úsek, ktorý X pokrýva. Zostávajúce kusy (≥ 1 m)
   ostanú s pôvodným typom. Inak by krátka X hrana zmazala dlhú polygónovú (Beluj R1: hrebeň
   14,5 m sa zmenil na 4,6 m, pokrytie obvodu 100 % → 67 %).
2. ŠTÍT NA CUDZEJ PRESNEJ HRANE: zvyšný `s` kus, ktorý leží na X hrane INEJ roviny, prevezme jej typ
   (spoločná hrana nie je štít).

Id hrán s predponou „X" sa považujú za presné (konvencia run_v3.py).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from app.core import contract, qa

MIN_PIECE_M = 1.0


def _is_x(e) -> bool:
    return str(e.id).startswith("X")


def _covered_intervals(e, xs, length: float) -> List[Tuple[float, float]]:
    (ax, ay), (bx, by) = qa._seg_xy(e)
    ux, uy = (bx - ax) / length, (by - ay) / length
    iv = []
    for x in xs:
        (x0, y0), (x1, y1) = qa._seg_xy(x)
        t0 = (x0 - ax) * ux + (y0 - ay) * uy
        t1 = (x1 - ax) * ux + (y1 - ay) * uy
        lo, hi = max(0.0, min(t0, t1)), min(length, max(t0, t1))
        if hi > lo:
            iv.append((lo, hi))
    iv.sort()
    merged: List[Tuple[float, float]] = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _pieces(e, covered, length: float, min_piece_m: float):
    """Úseky hrany mimo `covered` ako (t0, t1), kratšie než min_piece_m sa zahodia."""
    out, cur = [], 0.0
    for lo, hi in covered:
        if lo - cur >= min_piece_m:
            out.append((cur, lo))
        cur = max(cur, hi)
    if length - cur >= min_piece_m:
        out.append((cur, length))
    return out


def _sub_edge(e, t0: float, t1: float, length: float, suffix: str):
    f0, f1 = t0 / length, t1 / length
    lerp = lambda a, b, f: a + (b - a) * f
    start = [lerp(e.start[i], e.end[i], f0) for i in range(3)]
    end = [lerp(e.start[i], e.end[i], f1) for i in range(3)]
    return contract.EdgeRecord(id=f"{e.id}{suffix}", type=e.type,
                               length_m=round(math.hypot(end[0] - start[0], end[1] - start[1]), 3),
                               start=start, end=end, exact=e.exact)


def reconcile_edges(model, min_piece_m: float = MIN_PIECE_M) -> Dict[str, int]:
    """Očistí hrany v kontrakte (in-place). Vracia {'trimmed','dropped','retyped'}."""
    stats = {"trimmed": 0, "dropped": 0, "retyped": 0}

    # 1) duplicita s vlastnou X hranou → orezať, nie zmazať celé
    for p in model.planes:
        own_x = [e for e in p.edges if _is_x(e)]
        if not own_x:
            continue
        keep = []
        for e in p.edges:
            if _is_x(e):
                keep.append(e)
                continue
            hits = [x for x in own_x if qa._coincide(e, x)]
            if not hits:
                keep.append(e)
                continue
            (ax, ay), (bx, by) = qa._seg_xy(e)
            length = math.hypot(bx - ax, by - ay)
            if length < 1e-6:
                continue
            pieces = _pieces(e, _covered_intervals(e, hits, length), length, min_piece_m)
            if not pieces:
                stats["dropped"] += 1
                continue
            stats["trimmed"] += 1
            for k, (t0, t1) in enumerate(pieces):
                keep.append(_sub_edge(e, t0, t1, length, "" if len(pieces) == 1 else "abcdefgh"[k % 8]))
        p.edges = keep

    # 2) štít ležiaci na presnej hrane INEJ roviny prevezme jej typ
    x_edges = [(q.id, e) for q in model.planes for e in q.edges if _is_x(e)]
    for p in model.planes:
        used = {e.id for e in p.edges}
        for e in p.edges:
            if e.type != "s" or _is_x(e):
                continue
            for qid, x in x_edges:
                if qid != p.id and qa._coincide(e, x):
                    old = e.id
                    e.type = x.type
                    n = 1
                    while f"{x.type}{n}" in used:
                        n += 1
                    e.id = f"{x.type}{n}"
                    used.discard(old)
                    used.add(e.id)
                    stats["retyped"] += 1
                    break
    return stats
