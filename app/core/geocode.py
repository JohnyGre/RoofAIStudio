# -*- coding: utf-8 -*-
"""
Geocoding: adresa → GPS (WGS84) + S-JTSK transform.

Používa Nominatim (OpenStreetMap) s lokálnou cache.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

import requests

# Adresár pre cache (relatívne k projektu)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "cache")
_CACHE_FILE = os.path.join(_CACHE_DIR, "geocode.json")

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT = "RoofAIStudio/2.0 (roof analysis pipeline)"


def _load_cache() -> Dict[str, dict]:
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, dict]) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def geocode(address: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Adresa → GPS súradnice.

    Args:
        address: textová adresa (napr. "Átriová 9309/16, Trnava")
        use_cache: použiť lokálnu cache (default True)

    Returns:
        dict s kľúčmi: lat, lon, display_name, bbox (alebo None)
    """
    address = address.strip()
    if not address:
        return None

    cache = _load_cache()
    if use_cache and address in cache:
        return cache[address]

    params = {
        "q": address,
        "format": "json",
        "limit": 5,
        "addressdetails": 1,
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(_NOMINATIM_URL, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        print(f"geocode: Nominatim chyba: {e}")
        return None

    if not results:
        print(f"geocode: žiadny výsledok pre '{address}'")
        return None

    best = results[0]
    result = {
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
        "display_name": best.get("display_name", address),
        "bbox": best.get("boundingbox", []),
        "address_details": best.get("address", {}),
    }

    if use_cache:
        cache[address] = result
        _save_cache(cache)

    return result


def geocode_wgs84_to_sjtsk(lat: float, lon: float) -> Tuple[float, float]:
    """
    WGS84 (EPSG:4326) → S-JTSK / Krovak East-North (EPSG:5514).

    Args:
        lat, lon: geografické súradnice (WGS84)

    Returns:
        (x, y) v S-JTSK (metre)
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5514", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


if __name__ == "__main__":
    import sys

    addr = sys.argv[1] if len(sys.argv) > 1 else "Átriová 9309/16, Trnava"
    result = geocode(addr)
    if result:
        print(f"Adresa: {result['display_name']}")
        print(f"GPS:    {result['lat']:.6f}, {result['lon']:.6f}")
        try:
            x, y = geocode_wgs84_to_sjtsk(result["lat"], result["lon"])
            print(f"S-JTSK: {x:.2f}, {y:.2f}")
        except Exception as e:
            print(f"S-JTSK: pyproj nedostupný ({e})")
    else:
        print("Žiadny výsledok")
