# -*- coding: utf-8 -*-
"""
Ortofoto fetch: ZBGIS WMS (primárne) + OSM dlaždice (fallback).

ZBGIS WMS: https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get
OSM tiles: https://tile.openstreetmap.org/{z}/{x}/{y}.png
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import requests

_ZBGIS_WMS_URL = "https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get"
_OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_USER_AGENT = "RoofAIStudio/2.0 (roof analysis pipeline)"

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "ortho")


def _meters_per_degree_lat(lat: float) -> float:
    return 111132.92 - 559.82 * math.cos(2 * math.radians(lat)) + 1.175 * math.cos(4 * math.radians(lat)) - 0.0023 * math.cos(6 * math.radians(lat))


def _meters_per_degree_lon(lat: float) -> float:
    return 111412.84 * math.cos(math.radians(lat)) - 93.5 * math.cos(3 * math.radians(lat)) + 0.118 * math.cos(5 * math.radians(lat))


def make_bbox(lat: float, lon: float, extent_m: float = 80.0) -> Tuple[float, float, float, float]:
    """
    Vytvorí bounding box okolo GPS súradnice.

    Args:
        lat, lon: WGS84 súradnice stredu
        extent_m: veľkosť strany štvorca (m), default 80 m

    Returns:
        (xmin, ymin, xmax, ymax) v EPSG:3857 (Web Mercator)
    """
    # Prepočet na Web Mercator (EPSG:3857)
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0

    half = extent_m / 2.0
    xmin, xmax = x - half, x + half
    ymin, ymax = y - half, y + half
    return xmin, ymin, xmax, ymax


def fetch_zbgis_ortho(lat: float, lon: float, extent_m: float = 80.0, size: int = 4096) -> Optional[bytes]:
    """
    Stiahne ortofoto z ZBGIS WMS.

    Args:
        lat, lon: WGS84 súradnice stredu
        extent_m: veľkosť strany (m)
        size: rozlíšenie (px)

    Returns:
        JPEG bytes, alebo None pri chybe
    """
    xmin, ymin, xmax, ymax = make_bbox(lat, lon, extent_m)

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": "1",  # Ortofoto (názvy sú číselné v ZBGIS)
        "STYLES": "",
        "CRS": "EPSG:3857",
        "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
        "WIDTH": size,
        "HEIGHT": size,
        "FORMAT": "image/jpeg",
    }
    headers = {"User-Agent": _USER_AGENT}

    try:
        resp = requests.get(_ZBGIS_WMS_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        if len(resp.content) < 1000:
            print(f"zbgis: odpoveď príliš malá ({len(resp.content)} B) — asi chyba")
            return None
        return resp.content
    except Exception as e:
        print(f"zbgis: chyba: {e}")
        return None


def _tile_coords(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_osm_ortho(lat: float, lon: float, zoom: int = 19) -> Optional[bytes]:
    """
    Stiahne ortofoto z OSM dlaždíc (3×3 mriežka).

    Returns:
        PNG bytes, alebo None
    """
    from PIL import Image
    import io

    xtile, ytile = _tile_coords(lat, lon, zoom)
    headers = {"User-Agent": _USER_AGENT}

    try:
        tiles = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                url = _OSM_TILE_URL.format(z=zoom, x=xtile + dx, y=ytile + dy)
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                tiles.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))

        # Zlož 3×3 (256 px každá = 768 px)
        canvas = Image.new("RGB", (768, 768))
        for i, tile in enumerate(tiles):
            dx, dy = i % 3, i // 3
            canvas.paste(tile, (dx * 256, dy * 256))

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"osm: chyba: {e}")
        return None


def save_ortho(data: bytes, name: str, ext: str = "jpg") -> str:
    """Uloží ortofoto do data/ortho/ a vráti cestu."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    path = os.path.join(_DATA_DIR, f"{name}.{ext}")
    with open(path, "wb") as f:
        f.write(data)
    print(f"ortho: uložené → {path}")
    return path


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.core.geocode import geocode

    addr = sys.argv[1] if len(sys.argv) > 1 else "Átriová 9309/16, Trnava"
    geo = geocode(addr)
    if not geo:
        print("Geocoding zlyhal")
        sys.exit(1)

    lat, lon = geo["lat"], geo["lon"]
    print(f"GPS: {lat:.6f}, {lon:.6f}")

    safe = addr.replace("/", "_").replace(",", "").replace(" ", "_")

    # ZBGIS primárne
    jpg = fetch_zbgis_ortho(lat, lon)
    if jpg:
        save_ortho(jpg, f"{safe}_zbgis", "jpg")
    else:
        print("ZBGIS zlyhal → OSM fallback")
        png = fetch_osm_ortho(lat, lon)
        if png:
            save_ortho(png, f"{safe}_osm", "png")
