"""
Satellite Image Fetcher service for RoofAIStudio.
Given an address, geocodes it, then fetches a satellite view.

Backends (quality order):
1. playwright - Google Maps direct, zoom 22, anti-detection (best, free)
2. google - Google Static Maps API (requires key)
3. bing - 5x5 ESRI tile grid at zoom+1
4. osm - 3x3 ESRI tiles (fallback)
"""

import os
import time
import hashlib
import logging
import math
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

logger = logging.getLogger(__name__)


class SatelliteImageFetcher:
    """Fetches satellite imagery for a given address or coordinates."""

    NOMINATIM_USER_AGENT = "RoofAIStudio/1.0"

    def __init__(
        self,
        cache_dir: str = "data/satellite_cache",
        google_api_key: Optional[str] = None,
        image_size: int = 640,
        zoom: int = 21,
        backend: str = "playwright",
    ):
        self.cache_dir = cache_dir
        self.google_api_key = google_api_key
        self.image_size = min(image_size, 640)
        self.zoom = zoom
        self.backend = backend
        os.makedirs(cache_dir, exist_ok=True)
        self._geolocator = Nominatim(user_agent=self.NOMINATIM_USER_AGENT)
        self._last_geocode = 0

    # --- Public API ---

    def fetch_by_address(self, address, country=None, force_refresh=False):
        meta = {"address": address}
        lat, lon = self._geocode(address, country)
        if lat is None:
            return None, {"error": "Geocode failed", "address": address}
        return self._fetch_impl(lat, lon, meta, force_refresh)

    def fetch_by_coords(self, lat, lon, force_refresh=False):
        return self._fetch_impl(lat, lon, {"lat": lat, "lon": lon}, force_refresh)

    def _fetch_impl(self, lat, lon, meta, force_refresh):
        meta["lat"] = lat
        meta["lon"] = lon
        cache_key = self._cache_key(lat, lon)
        cache_path = os.path.join(self.cache_dir, cache_key)

        if not force_refresh and os.path.exists(cache_path):
            img = cv2.imread(cache_path)
            if img is not None:
                meta["source"] = "cache"
                return img, meta

        if self.backend == "playwright":
            img = self._fetch_playwright(lat, lon)
        elif self.backend == "google" and self.google_api_key:
            img = self._fetch_google(lat, lon)
        elif self.backend == "bing":
            img = self._fetch_tiles(lat, lon, grid=5, zoom_offset=1)
        else:
            img = self._fetch_tiles(lat, lon, grid=3, zoom_offset=0)
        meta["backend"] = self.backend

        if img is not None:
            cv2.imwrite(cache_path, img)
            meta["source"] = "fetched"
            meta["cache_path"] = cache_path
        return img, meta

    # --- Geocoding ---

    def _geocode(self, address, country=None):
        elapsed = time.time() - self._last_geocode
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        query = address
        if country:
            query = address + ", " + country
        try:
            location = self._geolocator.geocode(query, timeout=10)
            self._last_geocode = time.time()
            if location:
                return location.latitude, location.longitude
            return None, None
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            logger.error("Geocode error: %s", e)
            self._last_geocode = time.time()
            return None, None

    # --- Google Static Maps ---

    def _fetch_google(self, lat, lon):
        params = {
            "center": "{},{}".format(lat, lon),
            "zoom": str(self.zoom),
            "size": "{}x{}".format(self.image_size, self.image_size),
            "maptype": "satellite",
            "key": self.google_api_key,
            "scale": "2",
        }
        url = "https://maps.googleapis.com/maps/api/staticmap?" + urlencode(params)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error("Google Maps fetch failed: %s", e)
            return None

    # --- Playwright: Google Maps direct (anti-detection) ---

    def _fetch_playwright(self, lat, lon):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return self._fetch_tiles(lat, lon, grid=5, zoom_offset=1)

        zoom = min(self.zoom + 2, 22)
        maps_url = (
            "https://www.google.com/maps/@"
            + str(lat) + "," + str(lon) + "," + str(zoom) + "z"
            "/data=!3m1!1e3!5m1!1e4"
        )
        logger.info("Google Maps: (%.5f, %.5f) @ zoom=%d", lat, lon, zoom)

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                    ],
                )
                context = browser.new_context(
                    viewport={"width": 1024, "height": 1024},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                )
                page = context.new_page()
                page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => false})"
                )
                page.goto(maps_url, wait_until="load", timeout=30000)
                time.sleep(4)

                # Accept cookies (try multiple selectors)
                cookie_texts = ["Prija", "Accept"]
                for ct in cookie_texts:
                    for _ in range(2):
                        try:
                            btn = page.locator("button").filter(has_text=ct).first
                            if btn.is_visible(timeout=2000):
                                btn.click()
                                time.sleep(2)
                                break
                        except Exception:
                            pass
                        time.sleep(1)

                page.keyboard.press("Escape")
                time.sleep(2)

                # Google Maps UI cannot be removed via CSS (shadow DOM).
                # Instead: screenshot the full page, then center-crop.
                screenshot_bytes = page.screenshot(full_page=False)
                browser.close()

                arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    # Center-crop to remove Google Maps chrome
                    h, w = img.shape[:2]
                    cs = min(h, w)
                    y0 = (h - cs) // 2
                    x0 = (w - cs) // 2
                    img = img[y0:y0+cs, x0:x0+cs]
                    img = cv2.resize(img, (self.image_size, self.image_size))
                    logger.info("GMaps captured: %dx%d", img.shape[1], img.shape[0])
                return img
        except Exception as e:
            logger.error("Google Maps Playwright failed: %s", e)
            return None

    # --- Tile-based (ESRI) ---

    def _fetch_tiles(self, lat, lon, grid=5, zoom_offset=1):
        zoom = min(self.zoom + zoom_offset, 22)
        n = 2.0 ** zoom
        tile_x = int((lon + 180.0) / 360.0 * n)
        tile_y = int(
            (1.0 - math.log(
                math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))
            ) / math.pi) / 2.0 * n
        )
        tile_url = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )
        tile_size = 256
        tiles = []
        half = grid // 2

        for dy in range(-half, half + 1):
            row = []
            for dx in range(-half, half + 1):
                url = tile_url.format(z=zoom, y=tile_y + dy, x=tile_x + dx)
                try:
                    resp = requests.get(
                        url,
                        headers={"User-Agent": self.NOMINATIM_USER_AGENT},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    tile_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if tile_img is not None:
                        row.append(cv2.resize(tile_img, (tile_size, tile_size)))
                    else:
                        row.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
                except Exception:
                    row.append(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
            tiles.append(row)

        mosaic = np.vstack([np.hstack(row) for row in tiles])
        return cv2.resize(mosaic, (self.image_size, self.image_size))

    # --- Helpers ---

    def _cache_key(self, lat, lon):
        raw = "{:.5f}_{:.5f}_{}_{}".format(lat, lon, self.zoom, self.image_size)
        h = hashlib.md5(raw.encode()).hexdigest()
        return "sat_{}_{:.5f}_{:.5f}.jpg".format(h[:12], lat, lon)
