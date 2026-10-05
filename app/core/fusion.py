# -*- coding: utf-8 -*-
"""Fúzia zdrojov = jediné miesto, kde sa rozhoduje. Deterministické, logované.

Nahrádza tichú heuristiku z `fusion_pipeline.py`:
    area_final = cv_area if cv_area > 20 else lidar_area
ktorá bez pravidla a bez záznamu uprednostnila Vision aj vtedy, keď sa s LiDARom
rozišiel 2,6× až 68× (dôkaz: .cluster/output/fusion_results.json).

Pravidlo priority je per typ veličiny, nie globálne:
    pôdorys / obrys budovy  ->  GIS vektor   (hranica stavby je úradne daná)
    Z / sklon / plocha      ->  LiDAR        (klasifikované body, ±5 cm vo Z)
    kontrola / seed         ->  Vision       (nikdy nenesie geometriu hrany)

Každé rozhodnutie nesie protistranu a flag, takže sa dá spätne overiť.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ─── prahy a limity (deterministické, nie „odhad modelu") ─────────────────────
MIN_LIDAR_POINTS = 200          # pod tým je mračno pre plochu nedôstatočné
MIN_LIDAR_AREA_M2 = 20.0        # stará heuristika: pod tým sa použil CV
CONFLICT_REL_THRESHOLD = 0.35   # relatívny rozdiel, od ktorého ide o konflikt
MIN_PLANE_AREA_M2 = 1.0         # menšie plochy sú šum/parapety

PRIORITY: Dict[str, str] = {
    "planimetry": "gis",     # pôdorys / obrys
    "elevation": "lidar",    # Z
    "slope": "lidar",        # sklon
    "area": "lidar",         # plocha strechy
    "validation": "vision",  # len kontrola/seed
}

AUTHORITY_NOTE = {
    "gis": "G1 (vektorový obrys budovy) — autorita pôdorysu",
    "lidar": "L1 (klasifikované mračno, trieda 6) — autorita Z, sklonu a plochy",
    "vision": "V1 (YOLO segmentácia) — kontrola/seed, nikdy geometria hrany",
}


def lidar_is_trustworthy(lidar: Optional[Dict[str, Any]]) -> bool:
    if not lidar:
        return False
    pts = int(lidar.get("lidar_points") or 0)
    area = float(lidar.get("area_lidar_m2") or lidar.get("area_m2") or 0.0)
    return pts >= MIN_LIDAR_POINTS and area >= MIN_LIDAR_AREA_M2


def arbitrate(
    cv: Optional[Dict[str, Any]],
    lidar: Optional[Dict[str, Any]],
    gis: Optional[Dict[str, Any]] = None,
    min_area_m2: float = MIN_LIDAR_AREA_M2,
) -> Dict[str, Any]:
    """Vráti rozhodnutie per veličina + zoznam konfliktov a flagov.

    Vstup: slovníky, aké dnes produkujú `fusion_pipeline.py` / `cv_lidar_fusion.py`
    (`area_cv_m2`, `area_lidar_m2`, `lidar_points`, `pitch_deg`, ...).
    Nikdy nič „ticho" nevyberie: `area_final_m2` je nastavené len vtedy, keď je
    autorita dôveryhodná; inak je `None` a rozhodnutie ide do `flags`.
    """
    decisions: Dict[str, Any] = {}
    conflicts: List[Dict[str, Any]] = []
    flags: List[str] = []

    cv_area = float((cv or {}).get("area_cv_m2") or (cv or {}).get("area_final_m2") or 0.0) or None
    lid_area = float((lidar or {}).get("area_lidar_m2") or 0.0) or None
    gis_area = float((gis or {}).get("footprint_m2") or 0.0) or None
    pitch = (lidar or {}).get("pitch_deg", (cv or {}).get("pitch_deg"))

    # 1) KONFLIKT sa hlási vždy, keď sa zdroje rozchádzajú.
    if cv_area and lid_area:
        rel = abs(cv_area - lid_area) / max(cv_area, lid_area)
        if rel > CONFLICT_REL_THRESHOLD:
            conflicts.append({
                "aspect": "area",
                "vision_m2": round(cv_area, 2),
                "lidar_m2": round(lid_area, 2),
                "relative_diff": round(rel, 3),
                "resolved_by": "viz nižšie (priorita typov)",
            })

    # 2) Pôdorys: autorita GIS, ak existuje.
    if gis_area:
        decisions["footprint_m2"] = round(gis_area, 2)
        decisions["footprint_authority"] = AUTHORITY_NOTE["gis"]
    else:
        decisions["footprint_m2"] = None
        flags.append("NO_GIS_FOOTPRINT: chýba vektorový obrys — nesmie sa nahradiť bboxom")

    # 3) Plocha strechy: autorita LiDAR, ale len ak je dôveryhodný.
    if lidar_is_trustworthy(lidar):
        decisions["roof_area_m2"] = round(lid_area, 2)
        decisions["roof_area_authority"] = AUTHORITY_NOTE["lidar"]
    else:
        decisions["roof_area_m2"] = None
        pts = int((lidar or {}).get("lidar_points") or 0)
        flags.append(
            f"LOW_LIDAR_SUPPORT: {pts} bodov, lidar_area={lid_area} m² — plochu nehlásim ako autoritatívnu"
        )

    # 4) Sklon: autorita LiDAR.
    if pitch is not None:
        decisions["pitch_deg"] = round(float(pitch), 1)
        decisions["pitch_authority"] = AUTHORITY_NOTE["lidar"]

    # 5) Vision je vždy len indikatívny.
    if cv_area:
        decisions["vision_area_indicative_m2"] = round(cv_area, 2)
        decisions["vision_authority"] = AUTHORITY_NOTE["vision"]

    return {
        "decisions": decisions,
        "conflicts": conflicts,
        "flags": flags,
        "priority": dict(PRIORITY),
        "legacy_rule_replaced": "area_final = cv_area if cv_area > 20 else lidar_area",
    }


def legacy_area(record: Dict[str, Any]) -> float:
    """Presná replikácia starej heuristiky — len na porovnanie v QA reporte."""
    cv_area = float(record.get("area_cv_m2") or 0.0)
    lid_area = float(record.get("area_lidar_m2") or 0.0)
    return cv_area if cv_area > 20 else lid_area
