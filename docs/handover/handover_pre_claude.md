# RoofAIStudio — stav projektu a zadanie pre Claude (handover)

**Dátum:** 2026-10-06 · **Repo:** github.com/JohnyGre/RoofAIStudio (branch `master`) ·
**Kontrakt:** schema 3.0.0, CRS EPSG:8353 · **Stav:** CI OK, deterministické behy, QA 0 chýb

---

## 1. Čo je projekt

Automatické modelovanie striech: z adresy/GPS vyrobí **roviny strechy**, **klasifikované hrany**
(odkvap `o`, hrebeň `h`, nárožie `n`, úžľabie `u`, štít `s`), presné polygóny v S-JTSK a balík výstupov
(JSON kontrakt, DXF, HTML report, izometrický výkres SVG/PNG, GeoTIFF + GeoJSON). Beží lokálne,
deterministicky, bez LLM v geometrii.

### Kto je autorita pre čo

| Zdroj | Rola | Poznámka |
|---|---|---|
| LiDAR (LAZ, trieda 6) | **autorita geometrie** (roviny, sklony, výšky) | ZBGIS/MAPKA, JTSK03 + Bpv, cyklus 2018 (~5 b/m²) |
| GIS obrys budovy | planimetrická autorita (orezanie mračna) | OSM s geometriou; ZBGIS pripravený v `gis.zbgis_footprint` |
| Vycvičený model (YOLO-seg) | typológia + planimetrický podklad + QA | `ai_models/roof_gmaps_v2_last.pt`, 5 tried; IoU voči mračnu 0,76 |
| Ortofoto ZBGIS (WMS) | vizuálna kontrola a prekrytie | 4096 px, ≈0,7 cm/px |
| LLM | **nie je** v geometrii | len plán/QA/reportovanie |

## 2. Overené výsledky (posledné behy)

| Adresa | Rovín | Plocha (pôdorys) | Pôdorys | Pomer | Hrany o/h/n/u/s | QA |
|---|---|---|---|---|---|---|
| Átriová 7751/16A, Trnava | 11 | 287,03 m² | 343,69 m² | 0,84 | 10/8/14/15/12 | 0 chýb / 7 varovaní |
| Átriová 9309/16, Trnava | 4 | 131,06 m² | — | — | 4/3/6/2/4 | **0 chýb / 0 varovaní** |
| Beluj 50 | 9 | 186,05 m² | — | — | 5/6/13/6/11 | 0 chýb / 7 varovaní |

Skutočná (šikmá) plocha strechy je o +11 % (26°) až +16 % (30°) väčšia — kontrakt má `area_true_m2`,
QA to hlási ako `info`.

**Testy:** `test_core_contract.py` 13 OK · `test_edges_bridge.py` 16 OK · `test_edge_qa.py` 20 OK ·
regresia 3 adresy PASS · `tools\ci.cmd` → **CI OK**.

## 3. Spustenie

```powershell
py -m venv .venv && .venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe tools\run_v3.py --lat 48.3955928 --lon 17.5856480 --name Triova --ortho
tools\ci.cmd        # testy → selfcheck → QA → regresia → export; očakáva sa "CI OK"
```

## 4. Zadanie pre Clauda (v poradí priority)

### Úloha 1 — Doplniť odkvapové hrany (najbližšie)
QA hlási `plane_has_eave` (Triova 2×, Beluj 3×): rovinám po orezávaní zmizol ich vodorovný odkvap.
**Postup:** pre rovinu bez `o` nájsť na `engine.roof_outline_ring` úsek patriaci tejto rovine
(bod v polygóne, blízko roviny), vložiť ako `o` hranu s výškami z roviny v oboch koncoch.
**Kritérium:** `ci.cmd` OK; `plane_has_eave` = 0 na všetkých 3 adresách; všetky `o` ≤ 6°.

### Úloha 2 — Katastrálny obrys budovy (ZBGIS)
Dnes OSM (generalizovaný) alebo aproximácia. Verejný WFS 404 → overiť vrstvu alebo podporiť import
z MAPKA (SHP/GeoJSON/DXF). Miesto: `app/core/gis.py::zbgis_footprint()` (nikdy nerobí tichý fallback).
**Kritérium:** mračno orezané obrysom; `area_vs_footprint` v [0,55; 1,8]; odchýlka hrán vs. ortofoto < 0,3 m
(dnes 0,66–0,92 m).

### Úloha 3 — Inštančný segmentačný model
Súčasný model delí **typy**, nie roviny (fity na masky dávali sklony 25–65°). Postup bez externých dát:
vygenerovať tréningové dáta z overených v3 výstupov (masky v rastri ortofota — viď `preprocess.py`)
a dotrénovať YOLO-seg na inštancie (ultralytics + CUDA vo venv).
**Kritérium:** IoU inštancií ≥ 0,7; plochy s odchýlkou hrán < 0,3 m na 3 adresách.

### Úloha 4 — Zníženie fragmentácie rovín
Triova má 11 rovín (reálne ~6–8). Po úlohách 2–3 overiť; doladiť `merge_coplanar_planes`.
**Kritérium:** ≤ 8 rovín pri zachovaní 6 úžľabí / 4 hrebeňov (potvrdené používateľom).

## 5. Pravidlá (overené praxou)

- **LLM nesmie počítať geometriu** — čísla len v deterministickom kóde.
- **Odkvap je vodorovný** (≤ 6°); stúpajúca „najnižšia" hrana je štít.
- **Spoločná hrana nie je štít** a má rovnaký typ v oboch rovinách.
- **Žiadne duplicity** — kanonická je presná hrana X z priesečnice.
- **Susedné plochy zdieľajú hranu** (medzery > 5 cm = QA nález).
- `area_m2` je pôdorys; na materiál `area_true_m2`.
- **Zmena = zelené CI**: `tools\ci.cmd` → „CI OK".

## 6. Odkazy

- Vstupný bod: `README.md` → `README_V3.md`
- Testovacie adresy: `docs/v3/testovacie_adresy.md`
- Opravy podľa auditu: `docs/v3/opravy_podla_auditu.md`
- Vzorky výstupov: `docs/v3/priklady/`
- Commity: `06fc479` (QA z auditu) · `ab2b6c6` (opravy hrán) · `24dc7b2` (most hrán)
