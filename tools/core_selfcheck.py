#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core_selfcheck — overí nové jadro (kontrakt, fúzia, registrácia, QA) na reálnych dátach.

Spustenie (z koreňa projektu RoofAIStudio):
    .venv\\Scripts\\python.exe tools\\core_selfcheck.py

Výstup: `output/core_checks/report.json` + súhrn do konzoly.
Nič nemení na vstupných dátach — je to čítacia kontrola (fail-soft).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:  # UTF-8 vystup (Windows konzola byva cp1250)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, fusion, qa, registration  # noqa: E402

WS = Path(r"C:\Users\jangr\.openclaw-autoclaw\agents\roof\workspace")
OUT = ROOT / "output" / "core_checks"
OUT.mkdir(parents=True, exist_ok=True)

report = {"sections": {}}
print("=" * 78)
print("RoofAIStudio — kontrola jadra (kontrakt / fúzia / registrácia / QA)")
print("=" * 78)

# ─── 1. FÚZIA: porovnanie starej tichej heuristiky a nového pravidla ──────────
fr = WS / ".cluster" / "output" / "fusion_results.json"
if fr.exists():
    records = json.loads(fr.read_text(encoding="utf-8"))
    rows, new_authoritative, legacy_used_cv = [], 0, 0
    for i, r in enumerate(records, 1):
        dec = fusion.arbitrate(cv=r, lidar=r)          # GIS zatiaľ nemáme → flag
        legacy = fusion.legacy_area(r)
        auth = dec["decisions"].get("roof_area_m2")
        if auth is not None:
            new_authoritative += 1
        if legacy == (r.get("area_cv_m2") or 0):
            legacy_used_cv += 1
        rows.append({
            "i": i,
            "vision_m2": round(float(r.get("area_cv_m2") or 0), 2),
            "lidar_m2": round(float(r.get("area_lidar_m2") or 0), 2),
            "legacy_area_final_m2": round(float(legacy), 2),
            "new_roof_area_m2": auth,
            "conflicts": len(dec["conflicts"]),
            "flags": dec["flags"],
        })
    report["sections"]["fusion"] = {
        "records": len(records),
        "legacy_picked_vision": legacy_used_cv,
        "new_authoritative_area": new_authoritative,
        "detail": rows,
    }
    print(f"\n[1] FÚZIA — {len(records)} záznamov z fusion_results.json")
    print(f"    stará heuristika vybrala Vision v {legacy_used_cv}/{len(records)} prípadoch")
    print(f"    nové pravidlo označilo plochu za autoritatívnu v {new_authoritative}/{len(records)} prípadoch")
    for r in rows[:4]:
        print(f"    #{r['i']}: Vision {r['vision_m2']} m² vs LiDAR {r['lidar_m2']} m² "
              f"→ staré={r['legacy_area_final_m2']} m², nové={r['new_roof_area_m2']} "
              f"(konflikty={r['conflicts']}, flagy={len(r['flags'])})")
    ex = fusion.arbitrate(cv={"area_cv_m2": 644.79}, lidar={"area_lidar_m2": 245.4, "lidar_points": 7905, "pitch_deg": 13.3})
    print("    príklad rozhodnutia (Vision 644,79 / LiDAR 245,40, 7905 b):")
    print("      " + json.dumps(ex["decisions"], ensure_ascii=False))
    print("      konflikt: " + json.dumps(ex["conflicts"], ensure_ascii=False))
else:
    report["sections"]["fusion"] = {"error": f"chýba {fr}"}
    print("\n[1] FÚZIA — vstup nenájdený:", fr)

# ─── 2. KONTRAKT + QA na reálnych rovinách z projektu ────────────────────────
cand = [
    ROOT / "output" / "_triov__7751_16A__917_01_Trnava_meta.json",
    WS / ".cluster" / "DELIVERY" / "slnečná_988_60_917_01_trnava_meta.json",
]
meta_path = next((c for c in cand if c.exists()), None)
if meta_path:
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    model = contract.from_legacy_meta(meta, source_year=2018)
    errors = contract.validate(model)
    qa_res = qa.run_all_checks(model)
    report["sections"]["contract_qa"] = {
        "source_file": str(meta_path),
        "planes": len(model.planes),
        "schema_version": model.schema_version,
        "contract_errors": errors,
        "qa": qa_res,
    }
    print(f"\n[2] KONTRAKT + QA — {meta_path.name}")
    print(f"    rovín: {len(model.planes)}, schéma: {model.schema_version}, crs: {model.crs}")
    print(f"    chyby kontraktu: {len(errors)}")
    for e in errors[:5]:
        print("      -", e)
    print(f"    QA verdikt: {qa_res['verdict']} (errors={qa_res['counts']['errors']}, warnings={qa_res['counts']['warnings']})")
    for i in (qa_res["errors"] + qa_res["warnings"])[:5]:
        print(f"      [{i.get('severity')}] {i.get('check')}: {i.get('detail') or i.get('plane','')}")
else:
    report["sections"]["contract_qa"] = {"error": "nenašiel som žiadny meta.json"}
    print("\n[2] KONTRAKT + QA — vstup nenájdený")

# ─── 3. REGISTRÁCIA: CRS + rezíduá + časový nesúlad ─────────────────────────
try:
    # Referencia MUSÍ byť z tej istej adresy ako dokumentovaná hodnota:
    # pipeline_v2.md → Átriová 9309/16: GPS 48.395436, 17.586068 → S-JTSK -535663.66, -1256478.28
    gps = {"lat": 48.395436, "lon": 17.586068}
    conv = registration.normalize(float(gps["lat"]), float(gps["lon"]), "EPSG:4326")
    doc = (-535663.66, -1256478.28)
    try:
        from pyproj import Transformer
        _t14 = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
        _x14, _y14 = _t14.transform(float(gps["lon"]), float(gps["lat"]))
        conv_5514 = {"x": round(_x14, 3), "y": round(_y14, 3), "crs": "EPSG:5514"}
    except Exception:
        conv_5514 = None
    res = registration.residual_check([{"ref": doc, "src": (conv["x"], conv["y"])}])
    sources = [
        {"id": "lidar_lls", "role": "lidar", "acquired_year": 2018},
        {"id": "ortofoto_zbgis", "role": "vision", "acquired_year": 2024},
    ]
    reg = registration.registration_report(sources, [{"ref": doc, "src": (conv["x"], conv["y"])}])
    report["sections"]["registration"] = {"normalized": conv, "documented": doc, "residuals": res,
                                          "temporal_flags": reg["temporal_flags"], "pyproj": reg["pyproj_available"]}
    print("\n[3] REGISTRÁCIA")
    print(f"    WGS84({gps['lat']}, {gps['lon']}) → {conv['crs']} = ({conv['x']}, {conv['y']})")
    print(f"    dokumentovaná hodnota: {doc} → rezíduum mean={res['mean_m']} m, max={res['max_m']} m → {res['verdict']}")
    if conv_5514:
        res14 = registration.residual_check([{"ref": doc, "src": (conv_5514["x"], conv_5514["y"])}])
        print(f"    (EPSG:5514) {conv_5514['crs']} = ({conv_5514['x']}, {conv_5514['y']}) → rezíduum max={res14['max_m']} m → {res14['verdict']}")
        report.setdefault("sections", {}).setdefault("registration", {})["eps5514"] = {"conv": conv_5514, "residuals": res14}
    print(f"    pyproj dostupný: {reg['pyproj_available']}")
    for t in reg["temporal_flags"]:
        print(f"    TEMPORAL: {t['a']} vs {t['b']} → {t['gap_years']} rokov → {t['action']}")
except Exception as e:  # fail-soft, ale nahlas
    report["sections"]["registration"] = {"error": str(e)}
    print("\n[3] REGISTRÁCIA — chyba:", e)

# ─── 4. KONTRAKT: round-trip (JSON → objekt → JSON) ─────────────────────────
if meta_path:
    txt = model.to_json()
    back = contract.RoofModel.from_json(txt)
    rt_ok = len(back.planes) == len(model.planes) and back.schema_version == model.schema_version
    report["sections"]["roundtrip"] = {"ok": rt_ok, "bytes": len(txt.encode("utf-8"))}
    print(f"\n[4] ROUND-TRIP kontraktu: {'OK' if rt_ok else 'FAIL'} ({len(txt.encode('utf-8'))} B)")

(OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nReport: {OUT / 'report.json'}")
print("=" * 78)
