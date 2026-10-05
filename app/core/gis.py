# -*- coding: utf-8 -*-
"""GIS adaptér: vektorový obrys budovy ako autorita pôdorysu.

Priorita zdrojov pôdorysu (podľa rešerše):
  1. ZBGIS (kataster) — vektorová vrstva budov; vyžaduje overenie WFS vrstvy
     a licencie → `zbgis_footprint()` to hlási explicitne, nerobí tichý fallback.
  2. OSM (Overpass, viac mirror-ov) — dostupné hneď, vhodné do času ZBGIS vrstvy.
  3. Žiadny obrys → FLAG `NO_GIS_FOOTPRINT` (nikdy sa nenahrádza bboxom).

Fail-soft: keď sú siete nedostupné, použije sa cache; ak ani tá nie je, vráti sa
`footprint_m2 = None` a dôvod — beh pipeline pokračuje a QA to nahlási.
"""
from __future__ import annotations

import json
import math
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _post(url: str, data: bytes, timeout: int = 30):
    """POST s overeným SSL; pri expirovanom certifikáte skúsi neoverene (označí to)."""
    req = urllib.request.Request(url, data=data,
                                headers={"User-Agent": "RoofAIStudio/3.0 (footprint adapter)"})
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read()), False
    except ssl.SSLCertVerificationError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read()), True


def _area_m2(poly_lonlat) -> float:
    """Plocha polygónu v m² cez lokálnu rovinu (equirectangular) — pre malé objekty."""
    lat0 = sum(p[1] for p in poly_lonlat) / len(poly_lonlat)
    k = 111320.0
    kx = k * math.cos(math.radians(lat0))
    pts = [(lon * kx, lat * k) for lon, lat in poly_lonlat]
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _cache_path(lat: float, lon: float) -> Path:
    return CACHE_DIR / f"footprint_{lat:.6f}_{lon:.6f}.json"


def _cache_get(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    p = _cache_path(lat, lon)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["cached"] = True
            return d
        except Exception:
            return None
    return None


def _cache_put(lat: float, lon: float, data: Dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(lat, lon).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    except Exception:
        pass


def osm_footprint(lat: float, lon: float, radius_m: float = 40.0, timeout: int = 15,
                  cache_first: bool = True) -> Dict[str, Any]:
    """Obrys budovy z OSM (Overpass, viac mirror-ov) s cache a fail-soft správaním.

    Keď cache_first, a v cache je záznam s geometriou, live sťahovanie sa preskočí
    (sieť bola pri testovaní nestabilná: 504/timeout).
    """
    if cache_first:
        cached = _cache_get(lat, lon)
        if cached and cached.get("ring_lonlat"):
            cached.setdefault("note", "z cache")
            return cached
    query = f"""
    [out:json][timeout:{timeout}];
    way(around:{int(radius_m)},{lat},{lon})["building"];
    out geom;
    """
    body = urllib.parse.urlencode({"data": query}).encode()

    data, unverified, last_err = None, False, None
    for url in OVERPASS_MIRRORS:
        try:
            data, unverified = _post(url, body, timeout=timeout + 10)
            break
        except Exception as e:  # 504 / timeout / SSL — skús ďalší mirror
            last_err = f"{type(e).__name__}: {e}"
            continue

    if data is None:
        cached = _cache_get(lat, lon)
        if cached:
            cached["note"] = f"live fetch zlyhal ({last_err}); použitá cache"
            return cached
        return {"footprint_m2": None, "source": "none",
                "note": f"OSM nedostupné ({last_err}) a cache chýba → FLAG NO_GIS_FOOTPRINT"}

    best = None
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 4:
            continue
        ring = [(g["lon"], g["lat"]) for g in geom]
        a = _area_m2(ring)
        if best is None or a > best["area"]:
            best = {"area": a, "ring": ring, "osm_id": el.get("id")}

    if best is None:
        return {"footprint_m2": None, "source": "osm", "note": "žiadna budova v okolí (Overpass)"}

    res = {
        "footprint_m2": round(best["area"], 2),
        "source": "osm",
        "osm_id": best["osm_id"],
        "ring_lonlat": best["ring"],
        "ssl_unverified": unverified,
        "accuracy_note": "OSM obrys je generalizovaný (nie katastrálny) — do času ZBGIS vrstvy",
    }
    _cache_put(lat, lon, res)
    return res


def zbgis_footprint(lat: float, lon: float) -> Dict[str, Any]:
    """ZBGIS vektorový obrys budovy — zatiaľ neoverená vrstva.

    Zámerne NEROBÍ tichý fallback na bbox: vráti explicitnú chybu, aby nevzniklo
    to, čo dnes (footprint = bounding box 24,3 × 24,3 m pri streche 231 m²).
    """
    raise NotImplementedError(
        "ZBGIS vektorová vrstva budov nie je overená (WFS vrstva + licencia). "
        "Použi osm_footprint() ako náhradu a označ zdroj v kontrakte."
    )


def footprint_for(lat: float, lon: float, prefer: str = "osm") -> Dict[str, Any]:
    if prefer == "zbgis":
        try:
            return zbgis_footprint(lat, lon)
        except NotImplementedError as e:
            res = osm_footprint(lat, lon)
            res["note"] = f"ZBGIS nedostupný ({e}); použitý OSM"
            return res
    return osm_footprint(lat, lon)
