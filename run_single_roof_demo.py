"""
Single Roof Analyzer — Demo
============================
Spustenie: .venv/Scripts/python.exe run_single_roof_demo.py
Alebo z PyCharmu: pravy klik → Run
"""

import cv2
import os
from app.services.single_roof_analyzer import SingleRoofAnalyzer


def main():
    print("=" * 60)
    print("Single Roof Analyzer Demo")
    print("=" * 60)

    # 1. Init + load models
    print("\n[1/3] Loading SAM + YOLO models...")
    analyzer = SingleRoofAnalyzer(yolo_conf=0.25, min_mask_area=80)
    if not analyzer.load():
        print("FAILED to load models.")
        return
    print("      OK")

    # 2. Analyze an address
    ADDRESS = "Letecka 1, Bratislava"
    # Zmen adresu podla potreby:
    # ADDRESS = "Mikulasova 5, Kosice"

    print(f"\n[2/3] Analyzing: {ADDRESS}")
    result = analyzer.analyze_address(ADDRESS)

    if not result["success"]:
        print(f"      FAILED: {result.get('error', 'unknown error')}")
        return

    # 3. Show results
    best = result["best_plane"]
    print(f"\n[3/3] Results ({result['elapsed_s']:.1f}s, source={result['source']})")
    print(f"      Address:  {result['address']}")
    print(f"      GPS:      {result['lat']:.5f}, {result['lon']:.5f}")
    print(f"      Planes:   {result['total_planes']} total, showing top {len(result['planes'])}")
    print()

    for i, p in enumerate(result["planes"]):
        print(
            f"      [{i}] {p['class_name']:<14} "
            f"SAM={p['sam_score']:.3f}  "
            f"YOLO={p['yolo_score']:.2f}  "
            f"area={p['area_px']}px"
        )

    # Save visualization
    out_dir = "data/test_output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "single_roof_demo.jpg")
    cv2.imwrite(out_path, result["visualization"])
    print(f"\n      Visualization saved: {out_path}")

    # Also save raw satellite image
    sat_path = os.path.join(out_dir, "single_roof_satellite.jpg")
    cv2.imwrite(sat_path, result["image"])
    print(f"      Satellite image saved: {sat_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    analyzer.unload()


if __name__ == "__main__":
    main()
