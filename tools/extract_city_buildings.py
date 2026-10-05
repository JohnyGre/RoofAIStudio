# -*- coding: utf-8 -*-
"""
extract_city_buildings.py - Extrahuje OSM budovy pre viac slovenských miest z PBF.

Mestá -> Nominatim geocode -> bbox 6x6 km -> pyosmium filter (building=*) -> geojson.
"""
from __future__ import annotations

import json
import math
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

from app.core.geocode import geocode  # noqa: E402

CITIES = {
    "bratislava": "Bratislava, Slovensko",
    "nitra": "Nitra, Slovensko",
    "zilina": "Žilina, Slovensko",
    "kosice": "Košice, Slovensko",
    "presov": "Prešov, Slovensko",
    "banska_bystrica": "Banská Bystrica, Slovensko",
    "trencin": "Trenčín, Slovensko",
    "poprad": "Poprad, Slovensko",
}

PBF = os.path.join(PROJ, "data", "osm", "slovakia-latest.osm.pbf")
OUT_DIR = os.path.join(PROJ, "data", "osm")
HALF = 0.045  # ~5 km (1° lat = 111 km)


def city_center(name: str) -> tuple:
    g = geocode(name)
    if not g:
        raise RuntimeError(f"Geocode zlyhal: {name}")
    return float(g["lat"]), float(g["lon"])


def extract_buildings(lat: float, lon: float, half: float) -> list:
    import osmium

    minlat, minlon = lat - half, lon - half / max(math.cos(math.radians(lat)), 0.3)
    maxlat, maxlon = lat + half, lon + half / max(math.cos(math.radians(lat)), 0.3)

    class BHandler(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.features = []

        def way(self, w):
            if not w.is_closed():
                return
            if "building" not in w.tags:
                return
            try:
                coords = [(n.lon, n.lat) for n in w.nodes]
            except Exception:
                return
            if not coords:
                return
            clat = sum(c[1] for c in coords) / len(coords)
            clon = sum(c[0] for c in coords) / len(coords)
            if minlat <= clat <= maxlat and minlon <= clon <= maxlon:
                self.features.append({
                    "type": "Feature",
                    "properties": {
                        "osm_id": w.id,
                        "building": w.tags.get("building", "yes"),
                        "name": w.tags.get("name", ""),
                    },
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                })

    handler = BHandler()
    handler.apply_file(PBF, locations=True)
    return handler.features


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, query in CITIES.items():
        out = os.path.join(OUT_DIR, f"{key}_buildings.geojson")
        if os.path.exists(out):
            n = len(json.load(open(out, encoding="utf-8"))["features"])
            print(f"{key}: už existuje ({n} budov)")
            continue
        print(f"{key}: geocode...", flush=True)
        lat, lon = city_center(query)
        print(f"  centrum: {lat:.5f}, {lon:.5f}")
        feats = extract_buildings(lat, lon, HALF)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, ensure_ascii=False)
        print(f"  {len(feats)} budov -> {out}", flush=True)


if __name__ == "__main__":
    main()
