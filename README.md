# RoofAIStudio

Automatické modelovanie striech: **z adresy/GPS** vyrobí roviny strechy, klasifikované hrany
(odkvap / hrebeň / nárožie / úžľabie / štít), presné polygóny v S-JTSK a balík výstupov
(JSON kontrakt, DXF, HTML report, izometrický výkres, GeoTIFF, GeoJSON). Beží **lokálne** a **deterministicky**.

> **→ Kompletný sprievodca pre agentov a vývojárov: [README_V3.md](README_V3.md)**
> (architektúra, kto je autorita pre čo, spustenie, mapa súborov, schéma kontraktu, kritériá overenia)

## Stav v3 (overené)

| | |
|---|---|
| QA | **0 chýb / 0 varovaní** |
| Regresia (3 adresy) | **PASS** |
| Deterministický beh | dva po sebe idúce behy = identické výsledky |
| Príklad | Átriová 7751/16A, Trnava → **11 rovín, 328,9 m²** (pomer k pôdorysu **0,96**) |

## Rýchly štart

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt

.venv\Scripts\python.exe tools\run_v3.py --lat 48.39559280 --lon 17.58564796 --name Triova_7751_16A_Trnava --ortho
tools\ci.cmd          # testy → selfcheck → QA → regresia (3 adresy) → export   → "CI OK"
```

## Testovacie adresy

Zoznam overených adries (GPS, LAZ dlaždice, očakávané výsledky): [docs/v3/testovacie_adresy.md](docs/v3/testovacie_adresy.md).
Vzorky reálnych výstupov: [docs/v3/priklady/](docs/v3/priklady/) (kontrakty, QA reporty, regresný report, SVG výkres, GeoJSON masky).

## Štruktúra repozitára

```
app/core/        # jadro v3: contract, engine, fusion, gis, ortho, preprocess, qa, registration, vision
app/plugins/     # export kontraktu v3
tools/           # run_v3 (orchestrátor), regression, render_iso, v3_export, qa_gate, core_selfcheck, ci.cmd
tests/           # test_core_contract.py (13 kontrol)
docs/v3/         # doručená dokumentácia + vzorky výstupov
PIPELINE_V2.md   # pôvodná špecifikácia (v2) — historický kontext
```

## Zdroje dát a licencie

- **LiDAR (LAZ)** a **ortofoto**: ZBGIS / GKÚ Bratislava — CC BY 4.0 (pri publikácii uviesť zdroj).
- **Obrysy budov**: OpenStreetMap — ODbL.
- Natrénované modely (`ai_models/`) a mračná nie sú v repozitári (veľkosť); cesty k nim popisuje `README_V3.md`.
