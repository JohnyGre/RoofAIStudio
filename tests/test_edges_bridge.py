# -*- coding: utf-8 -*-
"""Test mostu hrán (desktop → kontrakt v3) na syntetických strechách.

Spustenie bez pytestu: python tests/test_edges_bridge.py

Strechy majú známu geometriu, takže vieme presne, aké hrany majú vzniknúť:
  * sedlová           → 1 hrebeň (h)
  * valbová           → 1 hrebeň (h) + 4 nárožia (n)
  * dve sedlové (T)   → aspoň 1 úžľabie (u) + hrebene (h)
Súradnice sú posunuté na reálny rád S-JTSK (~ -5.4e5, -1.3e6), aby test odhalil
numerické problémy s veľkými číslami.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, edges, qa  # noqa: E402

X0, Y0, Z0 = -541000.0, -1310000.0, 200.0
TAN = np.tan(np.radians(30.0))

ok = 0
fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  OK   {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {detail}")


def fit_lsq(points):
    """Rovnaké ako fit_lsq v roofai_desktop_v2.py (desktopový formát rovín)."""
    A = np.c_[points[:, 0], points[:, 1], np.ones(len(points))]
    coef, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    return n, coef


def desktop_planes(points, labels):
    out = []
    for lab in sorted(set(labels.tolist())):
        pts = points[labels == lab]
        n, coef = fit_lsq(pts)
        out.append({"n": n, "coef": coef, "pts": pts})
    return out


def grid(x0, x1, y0, y1, step=0.2, seed=1):
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel(), np.random.default_rng(seed)


def gable():
    x, y, rng = grid(0, 12, -4, 4)
    z = 5.0 - TAN * np.abs(y) + rng.normal(0, 0.01, x.size)
    lab = (y > 0).astype(int)
    return np.c_[x + X0, y + Y0, z + Z0], lab


def hip():
    x, y, rng = grid(0, 12, -4, 4)
    d = np.stack([x, 12 - x, y + 4, 4 - y])          # vzdialenosti k 4 stranám
    lab = np.argmin(d, axis=0)
    z = TAN * d.min(axis=0) + rng.normal(0, 0.01, x.size)
    return np.c_[x + X0, y + Y0, z + Z0], lab


def tee():
    """Hlavná sedlová (hrebeň || x) + krídlo (hrebeň || y) — vzniká úžľabie."""
    x, y, rng = grid(-2, 16, -4, 14)
    za = np.where((y > -4) & (y < 4), 5.0 - TAN * np.abs(y), -np.inf)
    zb = np.where((x > 4) & (x < 12) & (y > 0), 4.0 - TAN * np.abs(x - 8) + 0.0 * y, -np.inf)
    z = np.maximum(za, zb)
    keep = np.isfinite(z)
    x, y, z = x[keep], y[keep], z[keep]
    use_b = (np.maximum(za, zb)[keep] == zb[keep]) & (zb[keep] > za[keep])
    lab = np.where(use_b, np.where(x < 8, 2, 3), np.where(y > 0, 1, 0))
    z = z + rng.normal(0, 0.01, z.size)
    return np.c_[x + X0, y + Y0, z + Z0], lab


def run(points, labels):
    planes = desktop_planes(points, labels)
    eng = edges.desktop_planes_to_engine(planes)
    return planes, edges.exact_edges(eng)


print("sedlová strecha")
pts, lab = gable()
planes, ex = run(pts, lab)
types = sorted(e["type"] for e in ex)
check("presne 1 hrana", len(ex) == 1, str(types))
check("je to hrebeň (h)", types == ["h"], str(types))
if ex:
    e = ex[0]
    along_x = abs(e["end"][0] - e["start"][0]) > 10 and abs(e["end"][1] - e["start"][1]) < 0.3
    check("hrebeň beží pozdĺž x a má ~12 m", along_x and 10.5 < e["length_m"] < 12.5, f"{e['length_m']:.2f}")
    check("výška hrebeňa ~ 5 m nad základom", abs(e["start"][2] - (Z0 + 5.0)) < 0.15, f"{e['start'][2]:.2f}")

print("valbová strecha")
pts, lab = hip()
planes, ex = run(pts, lab)
types = sorted(e["type"] for e in ex)
check("1 hrebeň + 4 nárožia", types == ["h", "n", "n", "n", "n"], str(types))

print("dve sedlové (T)")
pts, lab = tee()
planes, ex = run(pts, lab)
types = sorted(e["type"] for e in ex)
check("existuje úžľabie (u)", "u" in types, str(types))
check("existuje hrebeň (h)", "h" in types, str(types))
check("žiadny neznámy typ", set(types) <= {"h", "n", "u"}, str(types))

print("zápis do kontraktu")
pts, lab = hip()
planes, _ = run(pts, lab)
model = contract.RoofModel(
    address="syntetická valba", gps={"lat": 48.0, "lon": 17.0},
    planes=[contract.PlaneRecord(id=f"R{i+1}", type="sedlová/valbová", pitch_deg=30.0, area_m2=a)
            for i, a in enumerate([40.0, 40.0, 20.0, 20.0])],
    sources=[contract.SourceRecord(id="lidar", role="lidar", crs="EPSG:8353", acquired_year=2018)],
    roof_area_m2=120.0,
)
# pred doplnením: QA hlási WARN (nemá hrany)
check("bez hrán: QA warning", any(i["check"] == "edge_types" for i in qa.check_edge_types(model)))
plane_areas = [{"src": i} for i in range(4)]
res = edges.add_edges_from_desktop(model, plane_areas, planes)
check("nájdených 5 hrán", res["edges_found"] == 5, str(res))
check("kontrakt validný", contract.validate(model) == [], str(contract.validate(model)))
check("po doplnení: QA bez warningu na hrany", not qa.check_edge_types(model))
check("každá hrana je pri oboch susedoch (10 záznamov)",
      sum(len(p.edges) for p in model.planes) == 10, str([len(p.edges) for p in model.planes]))
rt = contract.RoofModel.from_json(model.to_json())
check("round-trip JSON zachová hrany", sum(len(p.edges) for p in rt.planes) == 10)

print("odolnosť")
m2 = contract.RoofModel(address="x", gps={"lat": 0, "lon": 0}, planes=[
    contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=10.0)], sources=[])
r2 = edges.add_edges_from_desktop(m2, [{"slope": 30.0}], planes)         # bez kľúča 'src'
check("starý meta bez 'src' nehodí výnimku", r2["edges_attached"] == 0 and "note" in r2)
check("jedna rovina → žiadne hrany", edges.exact_edges(edges.desktop_planes_to_engine(planes[:1])) == [])

print(f"\nVYSLEDOK: {ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
