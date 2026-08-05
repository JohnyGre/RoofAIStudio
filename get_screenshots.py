import asyncio
import os
from pathlib import Path
from urllib.parse import quote
import logging
import time

from playwright.async_api import async_playwright
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import cv2
import numpy as np

# --- Konfigurácia ---
ADDRESSES = [
    "Átriová 7960/16M, 917 01 Trnava",
    "Andreja Kmeťa 1621/23, 917 01 Trnava",
    "svätého Cyrila a Metoda 3052/23, 917 01 Trnava",
]
OUTPUT_DIR = "batch_screenshots"
VIEWPORT_SIZE = {"width": 1280, "height": 1280}
ZOOM_LEVEL = 22 # Maximálny zoom
NOMINATIM_USER_AGENT = "RoofAIStudio/1.0"
IMAGE_SIZE = 1024 # Veľkosť výsledného obrázku po orezaní

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pomocné funkcie ---

def slugify(address: str) -> str:
    """Vytvorí bezpečný názov súboru z adresy."""
    return "".join(c for c in address.lower() if c.isalnum() or c in " _-").rstrip().replace(" ", "_")

async def geocode_address(address: str) -> tuple[float, float] | None:
    """Prevedie adresu na GPS súradnice."""
    try:
        geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)
        # Používame sync_to_async, pretože geopy je knižnica bez async podpory
        location = await asyncio.to_thread(geolocator.geocode, address, timeout=10)
        if location:
            logging.info(f"Geokódovanie úspešné pre '{address}': ({location.latitude}, {location.longitude})")
            return location.latitude, location.longitude
        else:
            logging.warning(f"Geokódovanie zlyhalo pre '{address}': Adresa nenájdená.")
            return None
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        logging.error(f"Chyba pri geokódovaní pre '{address}': {e}")
        return None
    except Exception as e:
        logging.error(f"Neočekávaná chyba pri geokódovaní pre '{address}': {e}")
        return None


async def get_map_screenshot(address: str, output_path: Path):
    """
    Vytvorí screenshot satelitnej mapy pre danú adresu s použitím presnej metódy
    z hlavnej aplikácie (geokódovanie + priama URL so súradnicami).
    """
    lat, lon = await geocode_address(address)
    if lat is None or lon is None:
        return

    maps_url = f"https://www.google.com/maps/@{lat},{lon},{ZOOM_LEVEL}z/data=!3m1!1e3"
    logging.info(f"Otváram URL: {maps_url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            viewport=VIEWPORT_SIZE,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => false})")

        try:
            await page.goto(maps_url, wait_until="load", timeout=60000)
            # Pevné čakanie, ako v pôvodnom kóde, aby sa mapa stihla načítať
            await asyncio.sleep(5)

            # Robustnejšie hľadanie cookies tlačidla
            cookie_texts = ["Prijať všetko", "Accept all", "Prijať", "Accept"]
            for text in cookie_texts:
                try:
                    # Používame flexibilnejší selektor
                    button = page.locator(f'button:has-text("{text}")').first
                    if await button.is_visible(timeout=2000):
                        logging.info(f"Našiel sa a kliká na cookies tlačidlo: '{text}'")
                        await button.click()
                        await asyncio.sleep(2) # Počkáme na spracovanie kliku
                        break
                except Exception:
                    pass # Tlačidlo sa nenašlo, skúsime ďalšie

            # Skrytie ostatných rušivých elementov
            await page.keyboard.press("Escape")
            await asyncio.sleep(1)


            logging.info("Vytváram screenshot celej stránky...")
            screenshot_bytes = await page.screenshot(full_page=False)

        finally:
            await browser.close()

    # Orezanie obrázku na stred, ako v pôvodnom kóde
    arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        # Orezanie na štvorec z menšej strany
        min_dim = min(h, w)
        y_offset = (h - min_dim) // 2
        x_offset = (w - min_dim) // 2
        cropped_img = img[y_offset:y_offset+min_dim, x_offset:x_offset+min_dim]

        # Zmena veľkosti na finálnu veľkosť
        resized_img = cv2.resize(cropped_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(output_path), resized_img)
        logging.info(f"Orezaný a zmenšený obrázok uložený do: {output_path}")
    else:
        logging.error("Nepodarilo sa dekódovať obrázok zo screenshotu.")


async def main():
    """Hlavná funkcia pre spustenie sťahovania."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    print(f"Výsledky sa uložia do priečinka: {output_dir.resolve()}")

    # Inštalácia prehliadačov pre Playwright, ak je to potrebné
    # os.system("playwright install")

    for address in ADDRESSES:
        logging.info("-" * 30)
        logging.info(f"Spracovávam adresu: {address}")
        slug = slugify(address)
        output_path = output_dir / f"{slug}.png"
        await get_map_screenshot(address, output_path)

if __name__ == "__main__":
    asyncio.run(main())