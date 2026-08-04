import asyncio
from pathlib import Path
import logging
import time

# Nastavenie cesty, aby sme mohli importovať z 'app'
import sys
sys.path.append(str(Path(__file__).parent.parent))

from app.services.single_roof_analyzer import SingleRoofAnalyzer
from app.core.logger import setup_logging

# --- Konfigurácia ---
TEST_ADDRESS = "Átriová 7960/16M, 917 01 Trnava"
OUTPUT_DIR = "data/test_analyzer_output"

# Nastavenie logovania
logger = setup_logging()
logging.getLogger('hpack').setLevel(logging.WARNING) # Zníženie ukecanosti z http2 knižnice

def run_test():
    """
    Spustí test novej verzie SingleRoofAnalyzer.
    """
    print("--- Spúšťam test pre SingleRoofAnalyzer v4 ---")
    
    # Vytvorenie výstupného priečinka
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"Výsledky sa uložia do: {output_path.resolve()}")

    # 1. Inicializácia a načítanie analyzátora
    print("\n1. Načítavam modely (YOLO+SAM)...")
    t0 = time.time()
    try:
        # Použijeme rovnaké parametre ako v UI
        analyzer = SingleRoofAnalyzer(yolo_conf=0.15, min_mask_area=80)
        if not analyzer.load():
            print("!!! Chyba: Nepodarilo sa načítať modely. Test sa ukončil.")
            return
    except Exception as e:
        print(f"!!! Kritická chyba pri inicializácii analyzátora: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"   -> Modely úspešne načítané za {time.time() - t0:.2f}s")

    # 2. Spustenie analýzy
    print(f"\n2. Spúšťam analýzu pre adresu: '{TEST_ADDRESS}'...")
    t1 = time.time()
    try:
        # Spustíme analýzu v debug móde, aby sa uložili aj medzikroky
        result = analyzer.analyze_address(TEST_ADDRESS, debug_mode=True)
    except Exception as e:
        print(f"!!! Kritická chyba počas analýzy adresy: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"   -> Analýza dokončená za {time.time() - t1:.2f}s")

    # 3. Spracovanie a zobrazenie výsledkov
    print("\n3. Výsledky analýzy:")
    if not result or not result.get("success"):
        error_msg = result.get('error', 'Neznáma chyba')
        print(f"   !!! Analýza zlyhala: {error_msg}")
        return

    total_planes = result.get("total_planes_center", 0)
    total_area = result.get("total_area_m2", 0)
    total_perimeter = result.get("total_perimeter_m", 0)

    print(f"   -> Adresa: {result.get('address')}")
    print(f"   -> Nájdených plôch (v centre): {total_planes}")
    print(f"   -> Celková plocha: {total_area:.2f} m²")
    print(f"   -> Celkový obvod: {total_perimeter:.2f} m")
    print(f"   -> Použitý backend: {result.get('source')}")
    print(f"   -> Kalibračný faktor: {result.get('px_per_m'):.2f} px/m")

    # 4. Uloženie vizualizácie
    vis_image = result.get("visualization")
    if vis_image is not None:
        import cv2
        # Prevod z BGR (OpenCV) na RGB pre správne uloženie
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        
        filename = f"vystup_{TEST_ADDRESS.replace(' ', '_').replace(',', '')}.png"
        output_file = output_path / filename
        try:
            cv2.imwrite(str(output_file), vis_image_rgb)
            print(f"\n4. Výsledná vizualizácia uložená do: {output_file}")
        except Exception as e:
            print(f"!!! Chyba pri ukladaní obrázku: {e}")
    else:
        print("\n4. Vo výsledkoch sa nenašla žiadna vizualizácia na uloženie.")

    print("\n--- Test dokončený ---")


if __name__ == "__main__":
    run_test()