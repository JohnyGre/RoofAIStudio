# -*- coding: utf-8 -*-
"""Roboflow Universe cez Playwright - detaily datasetov + download URL."""
import sys, os, json
sys.path.insert(0, r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio")

from playwright.sync_api import sync_playwright

datasets = [
    ("roof-labeling", "roof-segmentation-psn1f"),          # 8682 obr
    ("roof-segmentation-ddoyx", "house-segmentation-jmv74"),  # 1064 obr, farby
    ("vec-bvgxj", "roof-segmentation-zytzo"),              # 399 obr, 4 triedy
]

OUT = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\data\cache"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=[
        "--use-gl=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist",
    ])
    context = browser.new_context(
        viewport={"width": 1400, "height": 900},
        locale="en-US",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    )
    page = context.new_page()

    for user, project in datasets:
        url = f"https://universe.roboflow.com/{user}/{project}"
        print(f"\n=== {user}/{project} ===")
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(6000)
            # Cloudflare?
            title = page.title()
            print(f"  Title: {title[:80]}")
            if "Just a moment" in title or "Attention" in title:
                print("  ⚠️ Cloudflare challenge!")
                page.wait_for_timeout(10000)
                title = page.title()
                print(f"  Po čakaní: {title[:80]}")
            text = page.inner_text("body")[:800]
            print(f"  Text: {text[:400].replace(chr(10), ' | ')}")
            page.screenshot(path=os.path.join(OUT, f"rf_{project}.png"))
        except Exception as e:
            print(f"  EXC: {e}")

    browser.close()
