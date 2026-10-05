# -*- coding: utf-8 -*-
"""Verzovaný dátový kontrakt medzi krokmi pipeline (RoofAIStudio v3).

Prečo: dnešné kroky si odovzdávajú JSON bez verzie a bez pôvodu, takže dva behy
toho istého pipeline vyrobia rozdielne názvy kľúčov (napr. `height_ridge_m`
vs `height_above_ground_m`) a downstream sa nedá napojiť.

Tento modul definuje jedinú schému, ktorá nesie:
  * verziu schémy,
  * pôvod každého vstupu (zdroj, CRS, rok vzniku, licencia),
  * roviny a hrany v jednotnom tvare,
  * cieľový súradnicový systém (S-JTSK / JTSK03 + Bpv).

Autorita: geometriu zapisuje geometrický engine; LLM do týchto čísel nevstupuje.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "3.0.0"
TARGET_CRS = "EPSG:8353"  # S-JTSK / JTSK03 + Bpv (výšky)
CRS_ALIASES = {
    "EPSG:4326": "EPSG:4326",   # WGS84 (geokódovanie)
    "EPSG:3857": "EPSG:3857",   # Web Mercator (ZBGIS WMS dlaždice)
    "EPSG:5514": "EPSG:5514",   # S-JTSK / Krovak
    "EPSG:8353": "EPSG:8353",   # S-JTSK / JTSK03 + Bpv
    "JTSK": "EPSG:5514",
    "JTSK03": "EPSG:8353",
    "WGS84": "EPSG:4326",
}

EDGE_TYPES = {"o": "okap", "h": "hreben", "n": "narozie", "u": "uzlabie", "s": "stit"}
PLANE_TYPES = {"plochá", "sedlová", "valbová", "neurčitá",
               "valba (trojuholníková)", "vikier", "sedlová/valbová"}


@dataclass
class SourceRecord:
    """Pôvod jedného vstupu. Rok je povinný na detekciu časového nesúladu."""

    id: str
    role: str                      # "gis" | "lidar" | "vision"
    crs: str                       # kanonický tvar (EPSG:xxxx)
    acquired_year: Optional[int] = None
    license: Optional[str] = None
    note: str = ""

    def normalized_crs(self) -> str:
        return CRS_ALIASES.get(self.crs, self.crs)


@dataclass
class EdgeRecord:
    id: str
    type: str                      # o / h / n / u / s
    length_m: float
    start: List[float]
    end: List[float]
    exact: bool = False


@dataclass
class PlaneRecord:
    id: str
    type: str                      # plochá / sedlová / valbová / neurčitá
    pitch_deg: float
    area_m2: float = 0.0          # pôdorysný priemet
    area_true_m2: Optional[float] = None   # skutočná plocha strechy (pôdorys / cos sklonu)
    azimuth_deg: Optional[float] = None
    rmse_m: Optional[float] = None
    vertices: List[List[float]] = field(default_factory=list)
    edges: List[EdgeRecord] = field(default_factory=list)
    low_confidence: bool = False


@dataclass
class RoofModel:
    address: str
    gps: Dict[str, float]                    # {"lat":..., "lon":...}
    planes: List[PlaneRecord]
    sources: List[SourceRecord]
    ground_mnm: Optional[float] = None
    footprint_m2: Optional[float] = None     # pôdorys (autorita: GIS)
    roof_area_m2: Optional[float] = None     # plocha strechy (autorita: LiDAR)
    schema_version: str = SCHEMA_VERSION
    crs: str = TARGET_CRS
    created: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["created"] = self.created or time.strftime("%Y-%m-%d %H:%M:%S")
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "RoofModel":
        raw = json.loads(text)
        planes = []
        for p in raw.get("planes", []):
            edges = [EdgeRecord(**e) for e in p.get("edges", [])]
            planes.append(PlaneRecord(**{**p, "edges": edges}))
        sources = [SourceRecord(**s) for s in raw.get("sources", [])]
        base = {k: v for k, v in raw.items() if k not in ("planes", "sources")}
        return RoofModel(planes=planes, sources=sources, **base)


def validate(model: RoofModel) -> List[str]:
    """Vráti zoznam chýb kontraktu. Prázdny zoznam = model je v poriadku."""
    errors: List[str] = []
    if model.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version={model.schema_version!r} (očakávané {SCHEMA_VERSION!r})")
    if model.crs != TARGET_CRS:
        errors.append(f"crs={model.crs!r} (cieľový je {TARGET_CRS!r})")
    for s in model.sources:
        if s.normalized_crs() != TARGET_CRS and s.normalized_crs() not in CRS_ALIASES.values():
            errors.append(f"zdroj {s.id}: neznámy CRS {s.crs!r}")
        if s.acquired_year is None:
            errors.append(f"zdroj {s.id}: chýba acquired_year (nedá sa detegovať časový nesúlad)")
    if not model.planes:
        errors.append("model nemá ani jednu rovinu")
    for p in model.planes:
        if p.type not in PLANE_TYPES and not p.low_confidence:
            errors.append(f"rovina {p.id}: neznámy typ {p.type!r} (povolené: {sorted(PLANE_TYPES)})")
        if p.low_confidence and p.edges:
            errors.append(f"rovina {p.id}: low_confidence rovina nemá mať hrany")
        for e in p.edges:
            if e.type not in EDGE_TYPES:
                errors.append(f"hrana {e.id} (rovina {p.id}): neznámy typ {e.type!r}")
            if len(e.start) != 3 or len(e.end) != 3:
                errors.append(f"hrana {e.id}: očakávam 3D body (X, Y, Z)")
    return errors


# ─── Adaptéry na staré (legacy) výstupy ───────────────────────────────────────

def from_legacy_meta(meta: Dict[str, Any], source_year: Optional[int] = None) -> RoofModel:
    """Starý `*_meta.json` (project output) → kontrakt.

    Legacy schéma: {address, gps{lat,lon}, ground_mnm, area_m2, plast_m2, roviny[{slope, az, area_m2, rmse_m}]}
    """
    planes: List[PlaneRecord] = []
    for i, r in enumerate(meta.get("roviny", []), 1):
        slope = float(r.get("slope", 0.0))
        ptype = "plochá" if slope < 8.0 else ("sedlová" if 15.0 <= slope <= 45.0 else "valbová")
        planes.append(
            PlaneRecord(
                id=f"R{i}",
                type=ptype,
                pitch_deg=slope,
                area_m2=float(r.get("area_m2", 0.0)),
                azimuth_deg=r.get("az"),
                rmse_m=r.get("rmse_m"),
                vertices=[],
                edges=[],
                low_confidence=float(r.get("area_m2", 0.0)) < 1.0,
            )
        )
    gps = meta.get("gps", {})
    return RoofModel(
        address=meta.get("address", ""),
        gps={"lat": gps.get("lat"), "lon": gps.get("lon")},
        planes=planes,
        sources=[
            SourceRecord(id="lidar_legacy", role="lidar", crs="EPSG:8353",
                         acquired_year=source_year, license="CC BY 4.0",
                         note="legacy meta.json (bez explicitného roku)"),
        ],
        ground_mnm=meta.get("ground_mnm"),
        footprint_m2=None,
        roof_area_m2=meta.get("area_m2"),
    )


def from_legacy_planes(doc: Dict[str, Any], address: str = "") -> RoofModel:
    """Výstup `exact_roof_planes` → kontrakt (roviny + hrany s typmi)."""
    planes: List[PlaneRecord] = []
    for p in doc.get("planes", []):
        edges = [
            EdgeRecord(
                id=e.get("id", ""),
                type=e.get("type", "o"),
                length_m=float(e.get("length_m", 0.0)),
                start=list(e.get("start", [0, 0, 0])),
                end=list(e.get("end", [0, 0, 0])),
                exact=bool(e.get("exact", False)),
            )
            for e in p.get("edges", [])
        ]
        planes.append(
            PlaneRecord(
                id=p.get("id", ""),
                type="neurčitá" if p.get("low_confidence") else p.get("type", "neurčitá"),
                pitch_deg=float(p.get("pitch_deg", 0.0)),
                vertices=[list(v) for v in p.get("vertices", [])],
                edges=edges,
                low_confidence=bool(p.get("low_confidence", False)),
            )
        )
    return RoofModel(
        address=address,
        gps={"lat": doc.get("lat"), "lon": doc.get("lon")},
        planes=planes,
        sources=[
            SourceRecord(id="lidar_legacy", role="lidar", crs="EPSG:8353", acquired_year=2018,
                         license="CC BY 4.0", note="legacy planes JSON"),
        ],
    )
