#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qa_gate — brána kvality nad výstupom pipeline. Návratový kód 1 = neprešlo.

Spustenie:
    .venv\\Scripts\\python.exe tools\\qa_gate.py output\\_triov__7751_16A__917_01_Trnava_meta.json
    .venv\\Scripts\\python.exe tools\\qa_gate.py output\\x_roofmodel_v3.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:  # UTF-8 vystup (Windows konzola byva cp1250)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core import contract, qa  # noqa: E402


def load_model(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if "schema_version" in raw and "planes" in raw:
        return contract.RoofModel.from_json(json.dumps(raw, ensure_ascii=False))
    return contract.from_legacy_meta(raw, source_year=2018)


def main() -> int:
    if len(sys.argv) < 2:
        print("Použitie: python tools/qa_gate.py <meta.json|roofmodel_v3.json>")
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"CHYBA: neexistuje {path}")
        return 2

    model = load_model(path)
    errors = contract.validate(model)
    qa_res = qa.run_all_checks(model)

    print(f"QA gate: {path.name}")
    print(f"  schéma: {model.schema_version} | CRS: {model.crs} | rovín: {len(model.planes)} | zdrojov: {len(model.sources)}")
    print(f"  chyby kontraktu: {len(errors)}")
    for e in errors:
        print("    -", e)
    print(f"  QA: {qa_res['verdict']} (errors={qa_res['counts']['errors']}, warnings={qa_res['counts']['warnings']})")
    for i in qa_res["errors"] + qa_res["warnings"]:
        print(f"    [{i.get('severity')}] {i.get('check')}: {i.get('detail', i.get('plane', ''))}")

    out = path.with_name(path.stem + "_qa.json")
    out.write_text(json.dumps({"file": str(path), "contract_errors": errors, "qa": qa_res},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  report: {out}")
    return 1 if (errors or qa_res["errors"]) else 0


if __name__ == "__main__":
    sys.exit(main())
