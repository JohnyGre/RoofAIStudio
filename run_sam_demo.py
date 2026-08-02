"""
RoofAIStudio — SAM Demo Skript
================================
Spustenie z PyCharmu: pravým tlačidlom na tento súbor → Run 'run_sam_demo'
Alebo: .venv/Scripts/python.exe run_sam_demo.py

Čo robí:
1. Načíta RoofDetector (YOLO + SAM + OpenCV + línie)
2. Spustí YOLO-asistovanú SAM segmentáciu na testovacom obrázku
3. Uloží výsledný obrázok s polygónmi
"""

import cv2
import time
from app.ai.models.roof_detector import RoofDetector


def main():
    # 1. Inicializácia
    print("=" * 60)
    print("RoofAIStudio SAM Demo — RoofDetector v0.5.0")
    print("=" * 60)

    detector = RoofDetector()

    print("\n[1/4] Loading YOLO model (roof_finetuned.pt)...")
    detector.load(
        device="cpu",
        yolo_model_path="ai_models/roof_finetuned.pt",
    )
    print("      ✓ YOLO loaded (roof_finetuned.pt, mAP50 0.938)")

    # 2. Testovací obrázok
    # Použi vlastný obrázok — zmeň cestu podľa potreby
    IMAGE_PATH = "data/test_images/letecka_crop.jpg"
    # IMAGE_PATH = "data/test_images/aerial_house.jpg"  # alternatíva

    print(f"\n[2/4] Loading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"      ✗ Nenašiel som obrázok: {IMAGE_PATH}")
        print("      Skontroluj cestu alebo použi iný obrázok.")
        return
    print(f"      ✓ Loaded: {img.shape[1]}x{img.shape[0]} px")

    # 3. SAM auto segmentácia
    print("\n[3/4] Running YOLO → SAM auto segmentation...")
    t0 = time.time()
    result = detector.detect_sam(
        img,
        conf=0.25,       # YOLO confidence threshold (zníž pre viac detekcií)
        imgsz=640,       # resize na trénovaciu mierku
        min_mask_area=80,  # minimálna plocha masky v px
    )
    elapsed = time.time() - t0

    n = len(result["results"])
    print(f"      ✓ {n} roof planes segmented in {elapsed:.1f}s")
    for i, r in enumerate(result["results"][:8]):
        print(
            f"         [{i}] {r['class_name']:<14} "
            f"YOLO={r['score']:.2f}  SAM={r['sam_score']:.3f}  "
            f"area={r['area_640']}px"
        )
    if n > 8:
        print(f"         ... + {n - 8} more")

    # 4. Uloženie výsledku
    OUTPUT_PATH = "data/test_output/sam_demo_result.png"
    print(f"\n[4/4] Saving result: {OUTPUT_PATH}")
    import os

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, result["vis"])
    print(f"      ✓ Saved to {OUTPUT_PATH}")

    # Cleanup
    detector.unload()

    print("\n" + "=" * 60)
    print("DONE — otvor výsledný obrázok:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
