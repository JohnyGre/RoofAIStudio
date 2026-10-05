# -*- coding: utf-8 -*-
"""Overenie nového jadra bez pytest: python tests/test_core_contract.py

Kontroluje: validáciu kontraktu, pravidlá fúzie a QA na syntetických prípadoch
presne podľa reálnych zlyhaní projektu (slope_flat pri 36,9°, hrana bez typu,
low_conf rovina s hranami, konflikt Vision vs LiDAR).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:  # UTF-8 výstup (Windows konzola býva cp1250)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, fusion, qa  # noqa: E402

ok = 0
fail = 0


def check(name: str, cond: bool, detail: str = ""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  OK   {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {detail}")


def make_model(planes):
    return contract.RoofModel(
        address="test",
        gps={"lat": 48.0, "lon": 17.0},
        planes=planes,
        sources=[contract.SourceRecord(id="lidar", role="lidar", crs="EPSG:8353", acquired_year=2018)],
        roof_area_m2=100.0,
    )


print("kontrakt")
good = make_model([contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=100.0)])
check("validny model bez chyb", contract.validate(good) == [], str(contract.validate(good)))

bad_edge = make_model([contract.PlaneRecord(
    id="R1", type="sedlová", pitch_deg=30.0, area_m2=100.0,
    edges=[contract.EdgeRecord(id="o1", type="X", length_m=1.0, start=[0, 0, 0], end=[1, 0, 0])])])
check("neznámy typ hrany = chyba", any("neznámy typ" in e for e in contract.validate(bad_edge)))

bad_crs = make_model([])
bad_crs.crs = "EPSG:9999"
check("nespravny CRS = chyba", any("crs=" in e for e in contract.validate(bad_crs)))

check("round-trip JSON zachova roviny",
      len(contract.RoofModel.from_json(good.to_json()).planes) == 1)

print("QA")
flat_wrong = make_model([contract.PlaneRecord(id="R1", type="plochá", pitch_deg=36.9, area_m2=100.0)])
check("plochá pri 36,9° = error", any(i["check"] == "class_pitch" for i in qa.check_class_pitch(flat_wrong)))

low = make_model([contract.PlaneRecord(
    id="R1_LOW_CONFIDENCE", type="neurčitá", pitch_deg=10.0, area_m2=410.0, low_confidence=True,
    edges=[contract.EdgeRecord(id="o1", type="o", length_m=1.0, start=[0, 0, 0], end=[1, 0, 0])])])
issues = qa.check_low_confidence(low)
check("low_conf s hranami = error", any(i["severity"] == "error" for i in issues))
check("low_conf 410 m² = warning", any(i["severity"] == "warning" for i in issues))

gap_model = make_model([
    contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=50.0, vertices=[[0, 0, 0], [10, 0, 0], [10, 10, 0]]),
    contract.PlaneRecord(id="R2", type="sedlová", pitch_deg=31.0, area_m2=50.0, vertices=[[10.15, 0, 0], [20, 0, 0], [20, 10, 0]]),
])
check("medzera 0,15 m = warning", any(i["check"] == "gaps" for i in qa.check_gaps(gap_model)))

print("fúzia")
dec = fusion.arbitrate(cv={"area_cv_m2": 644.79}, lidar={"area_lidar_m2": 245.4, "lidar_points": 7905, "pitch_deg": 13.3})
check("konflikt sa nahlasi", len(dec["conflicts"]) == 1 and dec["conflicts"][0]["relative_diff"] > 0.35)
check("plocha = LiDAR (autorita)", dec["decisions"]["roof_area_m2"] == 245.4)
check("ziadny GIS obrys -> flag", any("NO_GIS_FOOTPRINT" in f for f in dec["flags"]))
weak = fusion.arbitrate(cv={"area_cv_m2": 606.02}, lidar={"area_lidar_m2": 8.83, "lidar_points": 502})
check("slabe mračno -> plocha None + flag",
      weak["decisions"]["roof_area_m2"] is None and any("LOW_LIDAR_SUPPORT" in f for f in weak["flags"]))
check("stara heuristika replikovana", fusion.legacy_area({"area_cv_m2": 644.79, "area_lidar_m2": 245.4}) == 644.79)

print(f"\nVYSLEDOK: {ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
