# Zdroje v v3 pipeline a ich roly (stav 2026-09-27)

## Čo je zapojené

| Zdroj | Rola podľa návrhu | Stav vo v3 | Dôkaz z posledného behu |
|---|---|---|---|
| **Point Cloud** (LAZ, trieda 6) | autorita Z / sklonu / plochy | **zapojené** | 12 266 bodov, po výškovom páse 11 838, po voxeli 10 656 |
| **GIS / ZBGIS** | autorita pôdorysu (vektorový obrys) | **čiastočne** — OSM obrys sa tentokrát podarilo získať **aj s geometriou** | orezanie obrysom: 10 656 → 10 240 bodov |
| **Vision (vycvičený model)** | planimetrický podklad + kontrola | **novo zapojené** (`app/core/vision.py`) | 12 masiek, triedy `slope_min=7, slope_tri=3, slope_poly=2`, obrys 377,3 m²; orezanie 10 240 → 10 170 |
| **Ortofoto (ZBGIS WMS)** | podklad pre kontrolu a prekrytie | **zapojené** | 4096 px, výrez 39,5 × 39,5 m (0,96 cm/px) |
| LLM | plán/QA (nikdy geometria) | zámerne **nezapojené** | — |

## Kvantitatívna zhoda Vision ↔ LiDAR

| Veličina | Hodnota |
|---|---|
| LiDAR obal (union polygónov rovín) | 346,8 m² |
| Vision obrys (segmentačný model) | 377,3 m² |
| Prienik | 265,0 m² |
| **IoU** | **0,577** |
| LiDAR mimo Vision | 81,8 m² |
| Vision neobsiahnuté LiDARom | 112,3 m² |

IoU 0,58 znamená, že zdroje si „sadajú" len čiastočne — je to merateľný ukazovateľ pre spoločný
registračný krok (co-registrácia ortofoto ↔ mračno), ktorý v projekte zatiaľ nie je.

## Vizuálna kontrola na ortofote (3 kolá)

| Kolo | Nález |
|---|---|
| 1 | odkvapy mimo obrysu aj vnútri strechy; „hrebeň" v skutočnosti úžľabie; úžľabia chýbali |
| 2 | po oprave konvexnosti: hrebene a nárožia zmizli úplne, úžľabia ako umelé obdĺžniky |
| 3 | odkvapy stále vedú naprieč strechou; úžľabia nesedia; hrebene a nárožia neoznačené |

## Diagnóza a ďalší krok

Roviny z RANSAC-u sú fragmenty (11 rovín, väčšina 10–52 m²), preto párové testy dávajú
nestabilné výsledky. Riešenie, na ktoré sú už pripravené všetky diely:

**Segmentácia riadená modelom** — použiť masky vycvičeného modelu (12 masiek) ako definíciu
strešných plôch a na body v každej maske fitovať rovinu (`app/core/vision.py` + `engine`).
Potom:
- roviny = fit na masky (nie RANSAC na celom mračne) → prestanú vznikať fragmenty,
- hrany = priesečnice susedných maskových rovín (presné čiary),
- odkvap = obrys zjednotenia masiek (nie obrys jednotlivých rovín).

Toto je presne princíp z rešerše: **hrany z obrazu, roviny z mračna**.
