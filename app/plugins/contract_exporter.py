# -*- coding: utf-8 -*-
"""Exportér verzovaného kontraktu v3 (`*_roofmodel_v3.json`).

Nahrádza dnešný stav, keď si každý krok písal vlastný JSON bez verzie a bez
pôvodu (dôkaz: dva behy mali rozdielne názvy kľúčov). Každý export nesie
schema_version, cieľový CRS, zoznam zdrojov s rokom vzniku a validáciu.

Použitie v kóde:
    from app.plugins.contract_exporter import export_contract, export_from_legacy_meta
    export_from_legacy_meta(meta_dict, 'output/x_roofmodel_v3.json')
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.core import contract, qa


def export_contract(model: "contract.RoofModel", out_path: str) -> str:
    """Zapíše model ako v3 kontrakt (JSON) a vráti cestu."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(model.to_json(), encoding="utf-8")
    return str(p)


def export_from_legacy_meta(
    meta: Dict[str, Any],
    out_path: str,
    sources: Optional[list] = None,
) -> Dict[str, Any]:
    """Legacy `*_meta.json` → v3 kontrakt + QA. Nič nemazá, len dopĺňa.

    Zdrojové roky sa musia uviesť explicitne (LiDAR LLS 1. cyklus 2017–2023,
    ZBGIS ortofotomozaika 3. cyklus 2023–2025), inak sa časový nesúlad nedá
    detegovať a kontrakt je neúplný.
    """
    model = contract.from_legacy_meta(meta)
    if sources:
        model.sources = [contract.SourceRecord(**s) if isinstance(s, dict) else s for s in sources]

    errors = contract.validate(model)
    qa_res = qa.run_all_checks(model)

    export_contract(model, out_path)
    qa_path = str(Path(out_path).with_name(Path(out_path).stem + "_qa.json"))
    Path(qa_path).write_text(
        json.dumps({"source": str(out_path), "contract_errors": errors, "qa": qa_res},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"contract_path": str(out_path), "qa_path": qa_path,
            "contract_errors": errors, "qa": qa_res}
