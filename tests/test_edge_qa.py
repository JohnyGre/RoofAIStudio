# -*- coding: utf-8 -*-
"""Test QA kontrol konzistencie hrán.

Spustenie bez pytestu: python tests/test_edge_qa.py

Fixture `roof_hip_valley_before_fix.json` je reálny výstup (anonymizovaný posunom
súradníc) z auditu — stav PRED opravou. Kontroly ho musia zachytiť. Syntetická
sedlová strecha so správnymi hranami musí prejsť bez nových varovaní (žiadne
falošné poplachy, ani na kolineárnych susedných štítoch).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, qa  # noqa: E402

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  OK   {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {detail}")


def E(id_, t, a, b, exact=True):
    return contract.EdgeRecord(id=id_, type=t, length_m=round(math.dist(a[:2], b[:2]), 3),
                               start=list(a), end=list(b), exact=exact)


# ---------------------------------------------------------------- reálna fixture
print("fixture z auditu (stav pred opravou)")
model = contract.RoofModel.from_json(
    (ROOT / "tests" / "fixtures" / "roof_hip_valley_before_fix.json").read_text(encoding="utf-8"))

eave = qa.check_eave_horizontal(model)
check("stúpajúci odkvap R1.o4 sa zachytí", any(i["plane"] == "R1" and i["edge"] == "o4" for i in eave), str(eave))
check("vodorovné odkvapy (R2.o3, R3.o1, R4.o1) nie sú označené",
      not any((i["plane"], i["edge"]) in {("R2", "o3"), ("R3", "o1"), ("R4", "o1")} for i in eave))

gs = qa.check_gable_shared(model)
planes_hit = {(i["plane"], i["edge"]) for i in gs}
check("štít R3.s2 na úžľabí R1/R3", ("R3", "s2") in planes_hit, str(planes_hit))
check("štít R4.s2 na nároží R3/R4", ("R4", "s2") in planes_hit, str(planes_hit))
check("pravý štít R1.s2 (voľná hrana) nie je označený", ("R1", "s2") not in planes_hit)
check("štít R2.s2 (voľná hrana) nie je označený", ("R2", "s2") not in planes_hit)

dup = qa.check_duplicate_edges(model)
txt = " | ".join(i["detail"] for i in dup)
check("duplicity: R3 hrebeň h3≈Xh2", "R3: hrany h3" in txt and "Xh2" in txt, txt)
check("duplicity: R4 nárožie n3≈Xn1", "R4: hrany n3" in txt and "Xn1" in txt, txt)
check("duplicita typov s2 vs o4 v R1", "R1: hrany s2" in txt and "o4" in txt, txt)
check("aspoň 6 duplicít (ako v audite)", len(dup) >= 6, str(len(dup)))
check("susedné kusy hrebeňa R2.h1 / R3.h3 nie sú duplicita",
      not any("h1" in i["detail"] and "h3" in i["detail"] for i in dup))

pe = qa.check_plane_has_eave(model)
check("R1 nemá vodorovný odkvap", [i["plane"] for i in pe] == ["R1"], str(pe))

at = qa.check_areas_true(model)
check("info o skutočnej ploche ~145,6 m²", len(at) == 1 and "145." in at[0]["detail"], str(at))
check("areas_true je len informácia (severity=info)", at and at[0]["severity"] == "info")

res = qa.run_all_checks(model)
check("verdikt WARN, 0 chýb", res["verdict"] == "WARN" and res["counts"]["errors"] == 0, str(res["counts"]))
check("info nezvyšuje počet varovaní", res["counts"]["info"] == 1)

# ---------------------------------------------------------------- čistá sedlová
print("čistá sedlová strecha (bez falošných poplachov)")
h = 4 * math.tan(math.radians(30))
X0, Y0 = -535000.0, -1256000.0   # reálny rád veľkosti


def P(x, y, z):
    return [X0 + x, Y0 + y, z]


A = contract.PlaneRecord(id="R1", type="sedlová", pitch_deg=30.0, area_m2=40.0, vertices=[
    P(0, 0, 0), P(10, 0, 0), P(10, 4, h), P(0, 4, h)], edges=[
    E("o1", "o", P(0, 0, 0), P(10, 0, 0), False), E("h1", "h", P(10, 4, h), P(0, 4, h)),
    E("s1", "s", P(10, 0, 0), P(10, 4, h), False), E("s2", "s", P(0, 4, h), P(0, 0, 0), False)])
B = contract.PlaneRecord(id="R2", type="sedlová", pitch_deg=30.0, area_m2=40.0, vertices=[
    P(0, 4, h), P(10, 4, h), P(10, 8, 0), P(0, 8, 0)], edges=[
    E("h1", "h", P(0, 4, h), P(10, 4, h)), E("s1", "s", P(10, 4, h), P(10, 8, 0), False),
    E("o2", "o", P(10, 8, 0), P(0, 8, 0), False), E("s2", "s", P(0, 8, 0), P(0, 4, h), False)])
clean = contract.RoofModel(address="x", gps={"lat": 0, "lon": 0}, planes=[A, B],
                           sources=[], roof_area_m2=80.0)
r = qa.run_all_checks(clean)
new_checks = {"eave_horizontal", "gable_shared", "duplicate_edges", "edge_type_consistency", "plane_has_eave"}
check("žiadne nové varovania na správnej streche",
      not [i for i in r["warnings"] + r["errors"] if i["check"] in new_checks],
      str([i["detail"] for i in r["warnings"]]))
check("áno, upozorní na pôdorysnú plochu (info)", [i["check"] for i in r["info"]] == ["areas_true"])

# ---------------------------------------------------------------- typová konzistencia
print("typová konzistencia spoločnej hrany")
bad = contract.RoofModel.from_json(clean.to_json())
bad.planes[1].edges[0].type = "n"            # R2.h1 označená ako nárožie, v R1 je hrebeň
tc = qa.check_edge_type_consistency(bad)
check("h vs n na tej istej čiare sa zachytí", len(tc) == 1, str(tc))

print("výšky odkvapu sa berú z roviny, nie z uložených z-hodnôt hrany")
bad2 = contract.RoofModel.from_json(clean.to_json())
bad2.planes[0].edges[0] = E("o1", "o", P(0, 0, 0), P(10, 0, 1.5), False)   # uložené z stúpa (8,5°), rovina je tam vodorovná
ev = qa.check_eave_horizontal(bad2)
check("odkvap, ktorý je v rovine vodorovný, sa neoznačí len kvôli zlému uloženému z", len(ev) == 0, str(ev))

print(f"\nVYSLEDOK: {ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
