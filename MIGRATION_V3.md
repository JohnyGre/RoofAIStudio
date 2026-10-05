# MIGRATION_V3 — čo sa zmenilo na základe rešerše

Táto zmena aplikuje **vlnu 1** (a časť vlny 2) odporúčania z architektonického reportu
(`RoofAIStudio_architektura.docx`) do reálneho projektu.

## 1. Nové jadro (`app/core/`)

| Súbor | Čo rieši | Odkiaľ to vzišlo |
|---|---|---|
| `app/core/contract.py` | **Verzovaný dátový kontrakt** (schema_version 3.0.0, cieľový CRS EPSG:8353, zoznam zdrojov s rokom vzniku, roviny a hrany v jednotnom tvare) + `validate()` | dôkaz E3: dva behy mali rozdielne názvy kľúčov (`height_ridge_m` vs `height_above_ground_m`) |
| `app/core/fusion.py` | **Deterministická fúzia** s prioritou per typ veličiny (pôdorys → GIS, Z/sklon/plocha → LiDAR, kontrola/seed → Vision), hlásenie konfliktov a flagov | dôkaz E1: `area_final = cv_area if cv_area > 20 else lidar_area` |
| `app/core/registration.py` | **Jeden cieľový CRS** pre všetky zdroje + kontrola rezíduí + detekcia časového nesúladu zdrojov | rešerš: WGS84 vs 3857 vs S-JTSK; ortofoto 2023–2025 vs LiDAR LLS 2017–2023 |
| `app/core/qa.py` | **QA kontroly**: trieda vs sklon, súčet plôch vs pôdorys, low_conf roviny, medzery medzi plochami, typy hrán | dôkaz E2 (`slope_flat` pri 36,9°) a low_conf rovina 410 m² |

## 2. Napojenie na existujúci kód

- `app/plugins/contract_exporter.py` — exportér `*_roofmodel_v3.json` (+ `*_qa.json`).
- `tools/qa_gate.py` — brána kvality s návratovým kódom (1 = neprešlo).
- `tools/core_selfcheck.py` — overenie jadra na reálnych dátach.
- `tests/test_core_contract.py` — 13 kontrol bez pytestu.
- `roofai_desktop_v2.py` → `_create_outputs()` — po zápise legacy `*_meta.json` sa
  automaticky zapíše aj **v3 kontrakt** a **QA report**. Celé je v `try/except`,
  takže beh GUI to nemôže zhodiť.

## 3. Overenie na reálnych dátach (spustené, nie teoretické)

```
.venv\Scripts\python.exe tests\test_core_contract.py         → 13 OK, 0 FAIL
.venv\Scripts\python.exe tools\core_selfcheck.py             → report v output\core_checks\report.json
.venv\Scripts\python.exe tools\qa_gate.py output\_triov__7751_16A__917_01_Trnava_roofmodel_v3.json
```

Výsledky:

| Kontrola | Výsledok |
|---|---|
| Stará heuristika (na 148 záznamoch fúzie) | vybrala Vision **148/148** prípadov |
| Nové pravidlo | označilo plochu za autoritatívnu (LiDAR) v **117/148**, zvyšok s flagom `LOW_LIDAR_SUPPORT` |
| Rekord #1 (644,79 vs 245,40 m²) | staré 644,79 m² → nové **245,40 m²** + konflikt (rel. rozdiel 0,619) |
| Rekord #3 (606,02 vs 8,83 m²) | staré 606,02 m² → nové **None** + 2 flagy (nedôstatočné mračno) |
| Registrácia (Átriová 9309/16) | rezíduum **0,091 m** (EPSG:8353), **0,034 m** (EPSG:5514) → OK |
| Časový nesúlad | LiDAR 2018 vs ortofoto 2024 → **6 rokov** → flag do QA |
| Kontrakt na reálnych rovinách | 22 rovín, 2 zdroje, schéma 3.0.0, **0 chýb** |
| QA | verdikt **WARN** (plochy bez klasifikovaných hrán — legacy meta ich neobsahuje) |
| Round-trip kontraktu | OK (5 583 B) |

## 4. Čo ešte zostáva (vlna 2 a 3)

1. **Vektorový obrys budovy z ZBGIS** namiesto bounding boxu — kontrakt je na to pripravený
   (`footprint_m2` + `NO_GIS_FOOTPRINT` flag), treba overiť dostupnosť vrstvy a licenciu.
2. **Krok registrácie pred enginom**: doplniť co-registráciu orto ↔ LiDAR ↔ GIS
   a ukladať rezíduá do kontraktu (modul `registration.py` je hotový, chýba napojenie na dáta).
3. **Klasifikácia hrán** do kontraktu: dnešné `*_meta.json` hrany neobsahujú, preto QA
   hlási WARN; napojiť `exact_roof_planes` výstup (id/typ/dĺžka/body hrany) na kontrakt.
4. **LLM vrstva až nakoniec** — a len ako router/QA reportér, nikdy nie ako sudca geometrie.
