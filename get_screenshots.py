import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright
import re

# --- Configuration ---
# Adresy, pre ktoré chceme získať snímky obrazovky
ADDRESSES_TO_CAPTURE = [
    "Átriová 7960/16M, 917 01 Trnava",
    "Andreja Kmeťa 1621/23, 917 01 Trnava",
    "svätého Cyrila a Metoda 3052/23, 917 01 Trnava",
]

OUTPUT_DIR = "batch_screenshots"
VIEWPORT_SIZE = {"width": 1280, "height": 1280}

def sanitize_filename(address):
    """Vytvorí bezpečný názov súboru z adresy."""
    # Odstráni diakritiku a nahradí medzery a špeciálne znaky podčiarkovníkom
    s = address.lower()
    s = ''.join(c for c in s if c.isalnum() or c.isspace() or c == '-')
    s = re.sub(r'\s+', '_', s)
    return s

async def get_map_screenshot(address: str, output_path: Path):
    """
    Spustí prehliadač, nájde adresu na Google Maps a urobí snímku obrazovky.
    """
    print(f"Spracovávam adresu: {address}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VIEWPORT_SIZE,
            device_scale_factor=2, # Pre vyššie rozlíšenie (Retina)
        )
        page = await context.new_page()

        try:
            # Otvoríme Google Maps
            await page.goto("https://www.google.com/maps", timeout=60000)

            # Súhlas s cookies, ak sa zobrazí
            try:
                await page.locator('button:has-text("Accept all")').click(timeout=5000)
                print("  -> Súhlas s cookies udelený.")
            except Exception:
                print("  -> Panel s cookies sa nezobrazil.")
                pass

            # Zadáme adresu do vyhľadávacieho poľa
            await page.locator("#searchboxinput").fill(address)
            await page.keyboard.press("Enter")
            print(f"  -> Vyhľadávam...")

            # Počkáme, kým sa mapa načíta a ustáli
            await page.wait_for_url(f"https://www.google.com/maps/place/**", timeout=30000)
            await asyncio.sleep(5) # Krátka pauza, aby sa všetko dokreslilo

            # Skryjeme bočný panel, aby sme mali plný pohľad na mapu
            try:
                await page.evaluate('''() => {
                    const sidePanel = document.querySelector('#QA0Szd');
                    if (sidePanel) sidePanel.style.display = 'none';
                }''')
                print("  -> Bočný panel skrytý.")
            except Exception:
                pass
            
            # Zmeníme na satelitný pohľad
            await page.locator("button[aria-label='Show satellite imagery']").click()
            print("  -> Prepol na satelitný pohľad.")
            await asyncio.sleep(3) # Pauza na načítanie satelitných snímok

            # Urobíme snímku obrazovky
            await page.screenshot(path=output_path)
            print(f"  -> Snímka obrazovky uložená do: {output_path}")

        except Exception as e:
            print(f"  -> CHYBA pri spracovaní adresy {address}: {e}")
        finally:
            await browser.close()

async def main():
    """
    Hlavná funkcia, ktorá prejde zoznam adries a spustí pre ne sťahovanie.
    """
    output_dir_path = Path(OUTPUT_DIR)
    output_dir_path.mkdir(exist_ok=True)
    print(f"Výsledky sa uložia do priečinka: {output_dir_path.resolve()}")
    
    # Inštalácia Playwright prehliadačov, ak je to potrebné
    print("Kontrolujem inštaláciu Playwright... (môže chvíľu trvať pri prvom spustení)")
    os.system("playwright install")
    print("Inštalácia dokončená.")
    print("-" * 30)

    for address in ADDRESSES_TO_CAPTURE:
        filename = sanitize_filename(address) + ".png"
        output_path = output_dir_path / filename
        await get_map_screenshot(address, output_path)
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())