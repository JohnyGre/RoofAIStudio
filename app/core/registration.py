# -*- coding: utf-8 -*-
"""Registrácia zdrojov: jeden cieľový CRS + kontrola rezíduí + časový nesúlad.

Prečo: geokód dáva WGS84 (4326), ZBGIS WMS dlaždice 3857 a LAZ je S-JTSK/JTSK03
(8353) + Bpv. Ak každý zdroj vojde do enginu vo svojom systéme, hrany sa
neprekrývajú a „area_final" je nezmysel. Tento modul je JEDINÉ miesto, kde sa
prepočítava a hlási rozsah registrácie.

Tiež hlási časový nesúlad: ortofoto 2023/2024/2025 vs LiDAR LLS 1. cyklus
2017–2023 — pri rozdieli > MAX_SOURCE_YEAR_GAP rokov ide flag do QA (reportovať,
nie ticho vyhladiť).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except Exception:  # pragma: no cover - bez pyproj necháme fail-soft
    _HAS_PYPROJ = False

TARGET_CRS = "EPSG:8353"          # S-JTSK / JTSK03 + Bpv
MAX_SOURCE_YEAR_GAP = 2           # roky; viac = flag TEMPORAL_MISMATCH
RESIDUAL_WARN_M = 0.50            # meter; viac = flag REGISTRATION_RESIDUAL


def _norm_crs(crs: str) -> str:
    from app.core.contract import CRS_ALIASES
    return CRS_ALIASES.get(crs, crs)


def normalize(x: float, y: float, src_crs: str) -> Dict[str, float]:
    """Prepočíta bod do cieľového CRS. Bez pyproj vracia jasne označený stav."""
    src = _norm_crs(src_crs)
    if src == TARGET_CRS:
        return {"x": float(x), "y": float(y), "crs": TARGET_CRS, "transformed": False}
    if not _HAS_PYPROJ:
        raise RuntimeError(
            f"pyproj nie je dostupný — nemôžem prepočítať {src} -> {TARGET_CRS}; "
            "registrovať zdroje ručne je zakázané (hlavný zdroj systémových chýb)."
        )
    if src == "EPSG:4326":
        t = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
        ex, en = t.transform(float(y), float(x))  # (lon, lat) -> (x, y)
        return {"x": round(ex, 3), "y": round(en, 3), "crs": TARGET_CRS, "transformed": True}
    t = Transformer.from_crs(src, TARGET_CRS, always_xy=True)
    ex, en = t.transform(float(x), float(y))
    return {"x": round(ex, 3), "y": round(en, 3), "crs": TARGET_CRS, "transformed": True}


def residual_check(points: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Rezíduá po registrácii (vzdialenosti dvojíc referencia↔zdroj).

    `points` = zoznam {"ref": (x, y), "src": (x, y)}. Vracia mean/max a verdikt.
    """
    import math

    pts = list(points)
    if not pts:
        return {"count": 0, "mean_m": None, "max_m": None, "verdict": "NO_DATA"}
    dists = [
        math.hypot(p["src"][0] - p["ref"][0], p["src"][1] - p["ref"][1])
        for p in pts
    ]
    mean = sum(dists) / len(dists)
    mx = max(dists)
    verdict = "OK" if mx <= RESIDUAL_WARN_M else "RESIDUAL_EXCEEDED"
    return {
        "count": len(pts),
        "mean_m": round(mean, 3),
        "max_m": round(mx, 3),
        "threshold_m": RESIDUAL_WARN_M,
        "verdict": verdict,
    }


def temporal_check(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deteguje časový nesúlad medzi zdrojmi (napr. ortofoto 2024 vs LiDAR 2018)."""
    flags: List[Dict[str, Any]] = []
    items = [s for s in sources if s.get("acquired_year")]
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            gap = abs(int(a["acquired_year"]) - int(b["acquired_year"]))
            if gap > MAX_SOURCE_YEAR_GAP:
                flags.append({
                    "kind": "TEMPORAL_MISMATCH",
                    "a": f"{a.get('id')}({a['acquired_year']})",
                    "b": f"{b.get('id')}({b['acquired_year']})",
                    "gap_years": gap,
                    "action": "reportovať do QA, nevyhladzovať",
                })
    return flags


def registration_report(sources: List[Dict[str, Any]], residuals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "target_crs": TARGET_CRS,
        "residuals": residual_check(residuals or []),
        "temporal_flags": temporal_check(sources),
        "pyproj_available": _HAS_PYPROJ,
    }
