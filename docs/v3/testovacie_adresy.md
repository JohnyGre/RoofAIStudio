# Testovacie adresy v3 pipeline

Toto sú adresy, na ktorých je pipeline overená (regresia + QA). Pri práci s repozitárom
používaj tieto údaje — vnútorné názvy súborov sa mierne líšia od skutočných adries.

| # | Adresa (skutočná) | GPS (WGS84) | Súbory (vnútorný názov) | LAZ dlaždice | Výsledok |
|---|---|---|---|---|---|
| 1 | **Átriová 7751/16A**, Kopánka, 917 01 Trnava | 48.39559280, 17.58564796 | `Triova_7751_16A_Trnava` | `05_Trnava_18_246985_5365887`, `247065_5365631`, `247070_5366181` | 11 rovín, 328,9 m², pôdorys 343,7 m² (pomer 0,96), QA 0/0 |
| 2 | **Átriová 9309/16**, Kopánka, 917 01 Trnava | 48.39543600, 17.58606800 | `Atriova_9309_16_Trnava` | `05_Trnava_18_246985_5365887` a okolie | 4 roviny, 131,1 m², QA 0/0 |
| 3 | **Beluj 50**, okres Banská Štiavnica, 969 01 Beluj | 48.35196013, 18.89217342 | `Beluj_50` | `12_BanskaStiavnica_19_343204_5358572` a okolie | 13 rovín, 168,7 m², QA 0/0 |
| (4) | Átriová 7960/16M, Kopánka, 917 01 Trnava | 48.39535207, 17.58632917 | `_triov__7960_16M` (starší beh) | `05_Trnava_18_*` | staršie dáta, nepoužité v regresii |

## Dôležité poznámky

- **Adresa č. 1 sa v názvoch súborov volá „Triova"** — je to tá istá budova ako „Átriová 7751/16A"
  (skratka vznikla pri zakladaní projektu, adresa v kontrakte je správna: *7751/16A, Átriová, Kopánka, Trnava*).
- GPS súradnice sú zo ZBGIS/geokódovania v **S-JTSK [JTSK03] / Bpv**; prepočet je v pipeline
  cez `pyproj` (EPSG:4326 → EPSG:8353).
- LAZ dáta pre Trnavu sú **cyklus 2018** (~5 b/m²), pre Beluj tiež starší cyklus;
  novší LiDAR (2./3. cyklus) by zlepšil detaily (vikiere, komíny).
- Všetky adresy používajú rovnaké zdroje: LiDAR (autorita geometrie), OSM obrys (planimetria),
  vycvičený model `ai_models/roof_gmaps_v2_last.pt` (typológia + podklad), ZBGIS ortofoto (kontrola).

## Ako spustiť regresiu na týchto adresách

```powershell
.venv\Scripts\python.exe tools\regression.py
# kritériá: 0 chýb kontraktu · QA bez chýb · ≥3 roviny · ≥3 klasifikované hrany · pomer 0,55–1,8
```

Samostatný beh jednej adresy:

```powershell
.venv\Scripts\python.exe tools\run_v3.py --lat 48.39559280 --lon 17.58564796 --name Triova_7751_16A_Trnava --ortho
```
