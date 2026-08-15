# -*- coding: utf-8 -*-
"""
MAPKA: ucel select - klik force + vyber moznost + licencia + export.
"""
import sys, os
sys.path.insert(0, r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio")

from playwright.sync_api import sync_playwright

OUT_DIR = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\cache"
os.makedirs(OUT_DIR, exist_ok=True)

EMAIL = "jangrexa@gmail.com"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=[
        "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
    ])
    context = browser.new_context(viewport={"width": 1600, "height": 1000}, locale="sk-SK")
    page = context.new_page()

    api_calls = []
    def on_response(resp):
        url = resp.url
        if any(k in url.lower() for k in ["export", "laz", "order", "objed", "download", "mrac", "checkavail"]):
            api_calls.append((resp.status, resp.request.method, url[:300]))
    page.on("response", on_response)

    page.goto("https://zbgis.skgeodesy.sk/mapka/sk/teren", wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(6000)
    try:
        btn = page.get_by_role("button", name="Pokračovať").first
        if btn.is_visible(timeout=3000):
            btn.click(); page.wait_for_timeout(2000)
    except Exception:
        pass

    # Adresa
    search = page.locator("input[type='text']").first
    search.click(); search.fill("Átriová 9309")
    page.wait_for_timeout(2500)
    page.locator("text=/Átriová 9309/").first.click(timeout=5000)
    page.wait_for_timeout(4000)
    try:
        page.locator("button, [role='button']").filter(has_text="close").first.click(timeout=2000)
        page.wait_for_timeout(1500)
    except Exception:
        pass

    # Pribliz
    for i in range(5):
        try:
            page.get_by_role("button").filter(has_text="add").first.click(timeout=2000)
            page.wait_for_timeout(1500)
        except Exception:
            break

    # Export panel
    page.get_by_role("button").filter(has_text="menu").first.click(timeout=5000)
    page.wait_for_timeout(2500)
    page.locator("text=Export údajov").first.click(timeout=5000)
    page.wait_for_timeout(4000)

    # Mracno + email
    page.locator("text=Mračno bodov").first.click(timeout=3000)
    page.wait_for_timeout(1500)
    page.locator("input[type='email']").first.fill(EMAIL)
    print("1. Mracno + email OK")

    # Ucel select - force click
    sel = page.locator("mat-select[name='dataUse']").first
    sel.click(force=True, timeout=5000)
    page.wait_for_timeout(2500)
    print("2. Ucel select otvoreny (force)")

    # Moznosti
    options = page.locator("mat-option").all()
    print(f"3. Moznosti: {len(options)}")
    for i, opt in enumerate(options):
        try:
            t = opt.inner_text()
            print(f"   [{i}] '{t}'")
        except Exception:
            pass

    # Vyber moznost - prva s textom (okrem prazdnej)
    chosen = False
    for i, opt in enumerate(options):
        try:
            t = opt.inner_text().strip()
            if t:
                opt.click(timeout=3000)
                print(f"4. Vybrate: '{t}'")
                chosen = True
                page.wait_for_timeout(2000)
                break
        except Exception:
            continue
    if not chosen:
        print("4. Ziadna moznost nevybrata!")

    # Licencia checkbox
    try:
        cb = page.locator("input[type='checkbox']").first
        cb.check(force=True)
        print("5. Licencia zaškrtnutá")
        page.wait_for_timeout(1500)
    except Exception as e:
        print(f"5. Licencia zlyhala: {e}")

    page.screenshot(path=os.path.join(OUT_DIR, "mapka_15_ready.png"))

    # Export
    try:
        exp = page.locator("span.mdc-button__label:has-text('Export')").first
        if exp.is_enabled(timeout=3000):
            exp.click(timeout=10000)
            print("6. EXPORT kliknuty!")
            page.wait_for_timeout(6000)
        else:
            print("6. Export stale disabled")
    except Exception as e:
        print(f"6. zlyhalo: {e}")

    page.screenshot(path=os.path.join(OUT_DIR, "mapka_16_done.png"))

    text = page.inner_text("body")[-2000:]
    print("\n=== KONIEC TELA ===")
    print(text)

    print("\n=== API ===")
    for status, method, url in api_calls[-15:]:
        print(f"  {status} {method} {url}")

    browser.close()
