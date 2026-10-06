# -*- coding: utf-8 -*-
"""Test finálnej očisty hrán (reconcile_edges). Spustenie: python tests/test_reconcile.py

Kľúčový prípad z reálneho behu: inline blok „4c" zmazal 14,5 m polygónový hrebeň Beluj R1,
lebo na ňom ležala 4,6 m X hrana (pokrytie obvodu 100 % → 67 %). Správne je hranu ORIEZNUŤ.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, eaves, qa, reconcile  # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  OK   {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {detail}")


def load(name):
    d = json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))
    try:
        return contract.RoofModel.from_json(json.dumps(d))
    except TypeError:
        for pl in d["planes"]:
            pl.pop("area_true_m2", None)
        return contract.RoofModel.from_json(json.dumps(d))


def plane(m, pid):
    return next(p for p in m.planes if p.id == pid)


def coverage(p):
    """Podiel obvodu polygónu pokrytý hranami (rovnobežné, do 0,6 m; zjednotenie úsekov)."""
    n, tot, cov = len(p.vertices), 0.0, 0.0
    for i in range(n):
        a, b = p.vertices[i], p.vertices[(i + 1) % n]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L < 1e-6:
            continue
        seg = contract.EdgeRecord(id="t", type="?", length_m=L, start=list(a), end=list(b), exact=False)
        iv = []
        for e in p.edges:
            if qa._coincide(seg, e):
                (x0, y0), (x1, y1) = qa._seg_xy(e)
                ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
                t = sorted([(x0 - a[0]) * ux + (y0 - a[1]) * uy, (x1 - a[0]) * ux + (y1 - a[1]) * uy])
                iv.append((max(0.0, t[0]), min(L, t[1])))
        iv.sort()
        total, cur = 0.0, 0.0
        for lo, hi in iv:
            if hi > max(lo, cur):
                total += hi - max(lo, cur)
                cur = hi
        tot += L
        cov += total
    return cov / tot if tot else 1.0


def E(i, t, a, b, exact=False):
    return contract.EdgeRecord(id=i, type=t, length_m=round(math.dist(a[:2], b[:2]), 3),
                               start=list(a), end=list(b), exact=exact)


X0, Y0 = -535000.0, -1256000.0
P = lambda x, y, z=0.0: [X0 + x, Y0 + y, z]


def one_plane(edges):
    pl = contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=40.0,
                              vertices=[P(0, 0, 0), P(20, 0, 0), P(20, 4, 2.3), P(0, 4, 2.3)], edges=edges)
    return contract.RoofModel(address="x", gps={"lat": 0, "lon": 0}, planes=[pl], sources=[], roof_area_m2=40.0)


print("polygónová hrana 14 m, X hrana 4 m uprostred → orezať, nie zmazať")
m = one_plane([E("h4", "h", P(3, 4, 2.3), P(17, 4, 2.3)), E("Xh1", "h", P(8, 4, 2.3), P(12, 4, 2.3), True)])
st = reconcile.reconcile_edges(m)
lens = sorted(round(e.length_m, 1) for e in m.planes[0].edges if not e.id.startswith("X"))
check("zostali 2 kusy 5 m + 5 m", lens == [5.0, 5.0], str(lens))
check("štatistika: 1 orezaná", st == {"trimmed": 1, "dropped": 0, "retyped": 0}, str(st))
check("bez duplicít podľa QA", qa.check_duplicate_edges(m) == [], str(qa.check_duplicate_edges(m)))
check("kusy zachovali typ aj výšky", all(e.type == "h" and abs(e.start[2] - 2.3) < 1e-6 for e in m.planes[0].edges))

print("X hrana pokrýva celú polygónovú → zmazať")
m = one_plane([E("n3", "n", P(2, 4, 2.3), P(10, 4, 2.3)), E("Xn1", "n", P(1.8, 4, 2.3), P(10.3, 4, 2.3), True)])
st = reconcile.reconcile_edges(m)
check("zmazaná (1 dropped), ostáva len X", st["dropped"] == 1 and [e.id for e in m.planes[0].edges] == ["Xn1"], str(st))

print("bez X hrán sa nič nemení; idempotentné")
m = one_plane([E("o1", "o", P(0, 0, 0), P(20, 0, 0))])
check("nič sa nezmení", reconcile.reconcile_edges(m) == {"trimmed": 0, "dropped": 0, "retyped": 0})
m = one_plane([E("h4", "h", P(3, 4, 2.3), P(17, 4, 2.3)), E("Xh1", "h", P(8, 4, 2.3), P(12, 4, 2.3), True)])
reconcile.reconcile_edges(m)
check("druhý beh nič nemení", reconcile.reconcile_edges(m) == {"trimmed": 0, "dropped": 0, "retyped": 0})

for fname, label in (("roof_11planes_before_eaves.json", "Triova (11 rovín)"),
                     ("roof_9planes_steep_before_eaves.json", "Beluj (9 rovín)")):
    print(f"reálny kontrakt: {label}")
    r = load(fname)
    eaves.quarantine_degenerate_planes(r)
    eaves.add_missing_eaves(r)
    before = {p.id: coverage(p) for p in r.planes if not p.low_confidence}
    st = reconcile.reconcile_edges(r)
    after = {p.id: coverage(p) for p in r.planes if not p.low_confidence}
    worst = min(after[k] - before[k] for k in before)
    check("pokrytie obvodu žiadnej roviny neklesne", worst > -0.02, f"{worst:.2f} {st}")
    res = qa.run_all_checks(r)
    check("QA: 0 chýb, 0 varovaní", res["counts"]["errors"] == 0 and res["counts"]["warnings"] == 0,
          str([(i['check'], i.get('plane')) for i in res['errors'] + res['warnings']]))
    check("idempotentné", reconcile.reconcile_edges(r) == {"trimmed": 0, "dropped": 0, "retyped": 0})
    if "9planes" in fname:
        r1 = plane(r, "R1")
        ridge = sum(e.length_m for e in r1.edges if e.type == "h")
        check("Beluj R1 zachoval hrebeň ~14,5 m (nie 4,6 m ako po inline bloku 4c)", ridge > 13.5, f"{ridge:.1f}")

print(f"\nVYSLEDOK: {ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
