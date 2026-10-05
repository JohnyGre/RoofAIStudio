#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""regression — spustí v3 pipeline na viacerých adresách a overí kritériá kvality.

Kritériá na adresu:
  * 0 chýb kontraktu,
  * QA bez chýb (varovania sú povolené, ale počítajú sa),
  * roviny >= 3,
  * pomer plocha/pôdorys v [0.55, 1.8] (ak poznáme pôdorys),
  * klasifikované hrany (h+n+u) >= 3.

Výstup: output/v3/regression_report.md + regression_results.json
Návratový kód 1 = aspoň jedna adresa nesplnila kritériá.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PY = str(P / ".venv" / "Scripts" / "python.exe")

ADDRESSES = [
    {"name": "Triova_7751_16A_Trnava", "lat": 48.39559280067209, "lon": 17.585647957122642, "ortho": True},
    {"name": "Atriova_9309_16_Trnava", "lat": 48.395436, "lon": 17.586068, "ortho": False},
    {"name": "Beluj_50", "lat": 48.35196012989301, "lon": 18.892173420481036, "ortho": False},
]


def run_addr(a: dict) -> dict:
    args = ["--lat", str(a["lat"]), "--lon", str(a["lon"]), "--name", a["name"]]
    if a.get("ortho"):
        args.append("--ortho")
    t0 = time.time()
    r = subprocess.run([PY, str(P / "tools" / "run_v3.py"), *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(P))
    dt = time.time() - t0
    ok = (r.returncode == 0)
    print(f"[{'OK' if ok else 'FAIL'}] {a['name']} ({dt:.0f}s)")
    if not ok:
        print(r.stdout[-1500:])
        print(r.stderr[-800:])
    return {"ok": ok, "seconds": round(dt, 1)}


def read_model(name: str):
    f = P / "output" / "v3" / f"{name}_roofmodel_v3.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8-sig"))


def read_qa(name: str):
    f = P / "output" / "v3" / f"{name}_qa.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text(encoding="utf-8-sig"))


def main() -> int:
    results = []
    for a in ADDRESSES:
        r = run_addr(a)
        m = read_model(a["name"]) or {}
        q = read_qa(a["name"])
        qa = q.get("qa", {})
        planes = m.get("planes", [])
        h = n = u = o = s = 0
        for pl in planes:
            for e in pl.get("edges", []):
                t = e.get("type")
                if t == "h": h += 1
                elif t == "n": n += 1
                elif t == "u": u += 1
                elif t == "o": o += 1
                else: s += 1
        roof = m.get("roof_area_m2") or 0
        foot = m.get("footprint_m2")
        ratio = (roof / foot) if (roof and foot) else None

        crit = {
            "contract_errors": len(q.get("contract_errors", [])),
            "qa_errors": len(qa.get("errors", [])),
            "qa_warnings": len(qa.get("warnings", [])),
            "planes": len(planes),
            "classified": h + n + u,
            "area_ratio": round(ratio, 2) if ratio else None,
            "pass": False,
        }
        crit["pass"] = (crit["contract_errors"] == 0 and crit["qa_errors"] == 0
                        and planes and len(planes) >= 3 and (crit["classified"] >= 3)
                        and (ratio is None or 0.55 <= ratio <= 1.8))
        results.append({"name": a["name"], "run_ok": r["ok"], "seconds": r["seconds"],
                        "roof_area_m2": roof, "footprint_m2": foot,
                        "edges": {"o": o, "h": h, "n": n, "u": u, "s": s}, **crit})

    # report
    md = ["# Regresný test v3 — výsledky", "",
          f"Beh: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
          "| Adresa | OK | s | Rovín | Plocha m² | Pôdorys m² | o/h/n/u/s | Kontr. ch. | QA chyb | QA var | Pomer | PASS |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    all_pass = True
    for r in results:
        e = r["edges"]
        line = (f"| {r['name']} | {'✓' if r['run_ok'] else '✗'} | {r['seconds']}s | {r['planes']} | "
                f"{r['roof_area_m2']} | {r['footprint_m2'] or '—'} | "
                f"{e['o']}/{e['h']}/{e['n']}/{e['u']}/{e['s']} | {r['contract_errors']} | "
                f"{r['qa_errors']} | {r['qa_warnings']} | {r['area_ratio'] or '—'} | "
                f"{'✓' if r['pass'] else '✗'} |")
        md.append(line)
        if not r["pass"]:
            all_pass = False
    md.append("")
    md.append(f"**Celkový verdikt: {'PASS' if all_pass else 'FAIL'}**")
    md.append("")
    md.append("Kritériá: 0 chýb kontraktu · QA bez chýb · ≥ 3 roviny · ≥ 3 klasifikované hrany · "
              "pomer plocha/pôdorys v [0,55; 1,8]")

    (P / "output" / "v3" / "regression_report.md").write_text("\n".join(md), encoding="utf-8")
    (P / "output" / "v3" / "regression_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md[2:]))
    print(f"\nReport: output/v3/regression_report.md")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
