# -*- coding: utf-8 -*-
"""Test doplnenia odkvapových hrán. Spustenie bez pytestu: python tests/test_eaves.py"""
from __future__ import annotations

import math
import sys
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, eaves, qa  # noqa: E402

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
    """Načíta fixture; ak starší contract.py nepozná area_true_m2, kľúč sa vynechá."""
    d = json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))
    try:
        return contract.RoofModel.from_json(json.dumps(d))
    except TypeError:
        for pl in d["planes"]:
            pl.pop("area_true_m2", None)
        return contract.RoofModel.from_json(json.dumps(d))


print("reálna fixture z auditu: R1 nemá skutočný odkvap")
m = load("roof_hip_valley_before_fix.json")
check("pred: plane_has_eave hlási R1", [i["plane"] for i in qa.check_plane_has_eave(m)] == ["R1"])
added = eaves.add_missing_eaves(m)
check("pridaný práve 1 odkvap, v R1", [(a["plane"], a["action"]) for a in added] == [("R1", "added")], str(added))
if added:
    r1 = next(p for p in m.planes if p.id == "R1")
    e = next(x for x in r1.edges if x.id == added[0]["edge"])
    zmin = min(v[2] for v in r1.vertices)
    check("je to nízka hrana (z ≈ 158 m), nie hrebeň (≈ 161 m)", abs(e.start[2] - zmin) < 0.3 and abs(e.end[2] - zmin) < 0.3,
          f"{e.start[2]:.2f} {e.end[2]:.2f}")
    check("dĺžka ~8,6 m", 8.0 < e.length_m < 9.2, str(e.length_m))
    check("vodorovný (≤ 6°) podľa QA", qa._eave_slope_deg(r1, e) <= qa.EAVE_MAX_DEG)
check("po: plane_has_eave = 0", qa.check_plane_has_eave(m) == [])
check("nová hrana nevyrobila duplicitu ani štít na cudzej hrane",
      not [i for i in qa.check_duplicate_edges(m) + qa.check_gable_on_shared_edge(m)
           if i.get("edge") == (added[0]["edge"] if added else None)])
check("idempotentné: druhé spustenie nič nepridá", eaves.add_missing_eaves(m) == [])

print("syntetická sedlová: odstránený odkvap sa obnoví")
h = 4 * math.tan(math.radians(30))
X0, Y0 = -535000.0, -1256000.0


def P(x, y, z):
    return [X0 + x, Y0 + y, z]


def E(i, t, a, b):
    return contract.EdgeRecord(id=i, type=t, length_m=round(math.dist(a[:2], b[:2]), 3),
                               start=list(a), end=list(b), exact=False)


def gable():
    A = contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=40.0,
                             vertices=[P(0, 0, 0), P(10, 0, 0), P(10, 4, h), P(0, 4, h)],
                             edges=[E("h1", "h", P(10, 4, h), P(0, 4, h)),
                                    E("s1", "s", P(10, 0, 0), P(10, 4, h)), E("s2", "s", P(0, 4, h), P(0, 0, 0))])
    B = contract.PlaneRecord(id="R2", type="sedlová", pitch_deg=30.0, area_m2=40.0,
                             vertices=[P(0, 4, h), P(10, 4, h), P(10, 8, 0), P(0, 8, 0)],
                             edges=[E("h1", "h", P(0, 4, h), P(10, 4, h)), E("o2", "o", P(10, 8, 0), P(0, 8, 0))])
    return contract.RoofModel(address="x", gps={"lat": 0, "lon": 0}, planes=[A, B], sources=[], roof_area_m2=80.0)


g = gable()
res = eaves.add_missing_eaves(g)
check("R1 dostala odkvap, R2 (už ho má) nie", [(a["plane"]) for a in res] == ["R1"], str(res))
e = g.planes[0].edges[-1]
check("odkvap leží na y=0 (nie na hrebeni y=4)", abs(e.start[1] - Y0) < 1e-6 and abs(e.end[1] - Y0) < 1e-6)
check("výšky z roviny, vodorovné", abs(e.start[2]) < 1e-6 and abs(e.end[2]) < 1e-6)
check("QA bez nových varovaní", not [i for i in qa.run_all_checks(g)["warnings"] if i["check"] in
                                        {"plane_has_eave", "eave_horizontal", "duplicate_edges", "gable_shared"}])

print("spoločná nízka hrana (úžľabie) nie je odkvap")
V1 = contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=40.0,
                          vertices=[P(0, 0, h), P(10, 0, h), P(10, 4, 0), P(0, 4, 0)], edges=[E("s1", "s", P(0, 0, h), P(0, 4, 0))])
V2 = contract.PlaneRecord(id="R2", type="sedlová", pitch_deg=30.0, area_m2=40.0,
                          vertices=[P(0, 4, 0), P(10, 4, 0), P(10, 8, h), P(0, 8, h)], edges=[E("s1", "s", P(10, 8, h), P(10, 4, 0))])
v = contract.RoofModel(address="x", gps={"lat": 0, "lon": 0}, planes=[V1, V2], sources=[], roof_area_m2=80.0)
check("V-strecha: žiadny odkvap sa nepridá", eaves.add_missing_eaves(v) == [])

def plane(m, pid):
    return next(p for p in m.planes if p.id == pid)


def eave_warnings(m):
    return [i["plane"] for i in qa.check_plane_has_eave(m) if i["severity"] == "warning"]


print("reálny kontrakt, 11 rovín (Triova): odkvap 6,1–6,2° bol štít; rovina s 0 m² tvorila falošné úžľabia")
t = load("roof_11planes_before_eaves.json")
check("pred: plane_has_eave varuje R4 a R5", sorted(eave_warnings(t)) == ["R4", "R5"], str(eave_warnings(t)))
check("pred: R4.s4 a R5.s4 sú štíty so sklonom 6,1–6,2°",
      all(plane(t, r).edges[k].type == "s" for r, k in (("R4", 2), ("R5", 1))))
quar = eaves.quarantine_degenerate_planes(t)
check("degenerovaná R1 (0 m²) sa vyradí", quar == ["R1"] and plane(t, "R1").low_confidence and not plane(t, "R1").edges, str(quar))
done = eaves.add_missing_eaves(t)
check("R4 aj R5 dostali odkvap povýšením štítu", sorted((d["plane"], d["action"]) for d in done) == [("R4", "promoted"), ("R5", "promoted")], str(done))
check("po: žiadne varovanie plane_has_eave", eave_warnings(t) == [], str(eave_warnings(t)))
check("R8 (malý vikier) ostáva len ako info", [i["severity"] for i in qa.check_plane_has_eave(t)] == ["info"])
check("povýšené odkvapy sú dlhé 11,2 m a 5,2 m",
      abs(next(e for e in plane(t, "R4").edges if e.type == "o").length_m - 11.19) < 0.05 and
      abs(next(e for e in plane(t, "R5").edges if e.type == "o").length_m - 5.23) < 0.05)
check("vodorovné 'úžľabia' z degenerovanej roviny zmizli", not [i for i in qa.check_edge_type_consistency(t)])
check("idempotentné", eaves.add_missing_eaves(t) == [] and eaves.quarantine_degenerate_planes(t) == [])

print("reálny kontrakt, 9 rovín (Beluj): odkvap vyhodený deduplikáciou kvôli sliveru 0,21 m²")
b = load("roof_9planes_steep_before_eaves.json")
check("pred: plane_has_eave varuje R4", eave_warnings(b) == ["R4"], str(eave_warnings(b)))
check("pred: R4 má vodorovné 'úžľabie' (sklon < 1°) na čiare odkvapu",
      any(e.type == "u" and qa._edge_slope_deg(e) < 1.0 for e in plane(b, "R4").edges))
quar = eaves.quarantine_degenerate_planes(b)
check("sliver R5 (0,21 m²) sa vyradí", quar == ["R5"], str(quar))
check("vodorovné 'úžľabie' R4 zmizlo spolu so sliverom",
      not any(e.type == "u" and qa._edge_slope_deg(e) < 1.0 for e in plane(b, "R4").edges))
done = eaves.add_missing_eaves(b)
acts = {(d["plane"], d["action"]) for d in done}
check("R4: odkvap pridaný, R9: štít povýšený", acts == {("R4", "added"), ("R9", "promoted")}, str(done))
r4o = next(e for e in plane(b, "R4").edges if e.type == "o")
check("R4 odkvap ~11,9 m a vodorovný", abs(r4o.length_m - 11.93) < 0.1 and qa._eave_slope_deg(plane(b, "R4"), r4o) < 2.0)
check("po: žiadne varovanie plane_has_eave", eave_warnings(b) == [], str(eave_warnings(b)))
check("R7 a R8 (malé strmé) iba info", sorted(i["plane"] for i in qa.check_plane_has_eave(b)) == ["R7", "R8"])

print("prah odkvapu je jeden (qa.EAVE_MAX_DEG) a medzi skutočnými odkvapmi (≤ 6,6°) a bočnicami (≥ 11°)")
check("8° leží v tejto medzere", 6.6 < qa.EAVE_MAX_DEG < 11.0, str(qa.EAVE_MAX_DEG))

print(f"\nVYSLEDOK: {ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
