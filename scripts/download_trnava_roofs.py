"""
Download 50 random roof images from Google Maps around Trnava.
Uses Playwright to fetch satellite view, center-crops, and saves.

Usage: python download_trnava_roofs.py

Output: data/gmaps_dataset/images/trnava_000.jpg ... trnava_049.jpg
         data/gmaps_dataset/trnava_metadata.jsonl
"""
import json, os, random, sys, time
import cv2, numpy as np
from datetime import datetime

# --- Trnava bounding box (approx, covers city + suburbs) ---
TRNAVA_BBOX = {
    "lat_min": 48.3600,
    "lat_max": 48.4000,
    "lon_min": 17.5550,
    "lon_max": 17.6200,
}

def random_coords(n=50):
    """Generate random GPS coordinates within Trnava."""
    coords = []
    for _ in range(n):
        lat = random.uniform(TRNAVA_BBOX["lat_min"], TRNAVA_BBOX["lat_max"])
        lon = random.uniform(TRNAVA_BBOX["lon_min"], TRNAVA_BBOX["lon_max"])
        coords.append((round(lat, 6), round(lon, 6)))
    return coords


def fetch_screenshot(playwright, lat, lon, out_path):
    """Fetch a single Google Maps satellite view and save."""
    try:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1024, "height": 1024})
        url = f"https://www.google.com/maps/@{lat},{lon},22z/data=!3m1!1e3!5m1!1e4"
        page.goto(url, wait_until="load", timeout=30000)
        time.sleep(5)

        # Dismiss cookie dialog
        try:
            for ct in ["Prijať", "Accept", "Accept all"]:
                btn = page.locator("button").filter(has_text=ct).first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    time.sleep(2)
                    break
        except Exception:
            pass

        page.keyboard.press("Escape")
        time.sleep(2)

        screenshot = page.screenshot()
        browser.close()

        # Decode and center-crop
        img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return False

        h, w = img.shape[:2]
        size = min(h, w)
        cy, cx = h // 2, w // 2
        half = size // 2
        crop = img[cy - half:cy + half, cx - half:cx + half]

        # Resize to 640x640
        final = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, final)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    os.makedirs("data/gmaps_dataset/images", exist_ok=True)

    coords = random_coords(50)
    metadata = []

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        for i, (lat, lon) in enumerate(coords):
            fname = f"trnava_{i:03d}.jpg"
            out_path = f"data/gmaps_dataset/images/{fname}"
            print(f"[{i+1}/50] {lat}, {lon}...", end=" ", flush=True)

            success = fetch_screenshot(p, lat, lon, out_path)

            if success:
                img = cv2.imread(out_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                metadata.append({
                    "filename": fname,
                    "lat": lat,
                    "lon": lon,
                    "zoom": 22,
                    "image_size": list(img.shape[:2]),
                    "mean_brightness": round(mean_brightness, 1),
                    "downloaded_at": datetime.now().isoformat(),
                    "skipped": False,
                })
                print(f"OK ({img.shape[1]}x{img.shape[0]}, brightness={mean_brightness:.0f})")
            else:
                metadata.append({
                    "filename": fname,
                    "lat": lat,
                    "lon": lon,
                    "skipped": True,
                    "downloaded_at": datetime.now().isoformat(),
                })
                print("SKIPPED")

            # Rate limit
            time.sleep(2)

    # Sort out skips
    success = sum(1 for m in metadata if not m.get("skipped"))
    skipped = sum(1 for m in metadata if m.get("skipped"))
    print(f"\nDone: {success} downloaded, {skipped} skipped")

    # Save metadata
    with open("data/gmaps_dataset/trnava_metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Also save a simple CSV for training
    with open("data/gmaps_dataset/train_images.txt", "w", encoding="utf-8") as f:
        for m in metadata:
            if not m.get("skipped"):
                f.write(f"images/{m['filename']}\n")

    print(f"Metadata saved to data/gmaps_dataset/")
    print(f"\nTrain with: yolo train data=data/gmaps_dataset/dataset.yaml ...")


if __name__ == "__main__":
    main()
