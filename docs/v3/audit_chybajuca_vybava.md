# Subagent 06 — Nezávislý audit: čo chýba do dokonalosti (RoofAIStudio)

## Overenie zoznamu (dôkazy z kódu a výstupov)

**1. Vektorový obrys ZBGIS — PLATÍ.** `app/core/gis.py::zbgis_footprint()` je `NotImplementedError` ("WFS vrstva + licencia neoverené"); beží len `osm_footprint()` s poznámkou "OSM obrys je generalizovaný (nie katastrálny)". `tools/run_v3.py:219` tlačí flag `FOOTPRINT_GEOMETRY_APPROXIMATED` a bez geometrie neorezáva mračno obrysom (použije celý klaster).

**2. Model na inštancie rovín — PLATÍ (čiastočne hotové, neintegrované).** `roofs_mega_v2`/`roofs_mega` majú **jedinú triedu `0: roof`** (data.yaml); labely obsahujú viac polygónov/obrázok (priemer ~2, max 29), takže ide o inštancie striech, nie overené inštancie rovín. Model `roof_mega_v2_s` je natrénovaný (task=segment, 20 epôch), ale **pipeline ho nepoužíva**: `run_v3.py MODEL = roof_gmaps_v2_last.pt` (5 typov slope_flat/min/poly/trap/tri, `vision.py`, `roof_type.py`).

**3. Co-registrácia orto↔mračno — PLATÍ.** `engine.register_offset()` je hrubá mriežka ±3 m, krok 0,25 m, max IoU (docstring: namerané IoU 0,58; výstup `*_pred_registraciou.json`). `app/core/registration.py` rieši len CRS/rezíduum/časový nesúlad; MIGRATION_V3 priznáva "chýba napojenie na dáta". Submeterové zarovnanie hrán na ortofoto nie je dosiahnuté (experiment 0,52→0,75 m).

**4. Odkvap ako uzavretý obvod — PLATÍ.** V poslednom behu má R1 7 úsekov odkvapu (o1–o7) + 3 úžľabia; okrem toho duplicitné X-hrany (Xu1–Xu3) prekrývajú tie isté línie. QA nekontroluje uzavretosť odkvapového okruhu.

**5. Topológia bez medzier — PLATÍ.** QA posledného behu: presne 4 varovania `gaps` 0,093–0,221 m (R2R5, R3R5, R4R9, R8R10), `GAP_TOL_M = 0.05` (`qa.py`).

**6. Zobrazenie v3 v GUI — PLATÍ.** `app/ui/panels/` je **prázdny adresár**. Desktop (`roofai_desktop_v2.py`) v3 kontrakt+QA len zapíše fail-soft do súboru a vypíše jeden log-riadok "(QA: WARN)"; hrany, flagy a konflikty sa nezobrazujú. 3D viewer sa otvára externe cez `os.startfile` (HTML).

**7. Viac adries — PLATÍ.** V `output/` len Triová 7751/16A Trnava a Beluj 50 (+ Atriová test), teda 2–3 budovy.

**8. Regresné testy + CI — PLATÍ.** `tests/test_core_contract.py` (13 kontrol, vlastný runner "bez pytestu"), `tools/qa_gate.py`, `tools/core_selfcheck.py` a `spustit.cmd vsetko` existujú, ale **žiadny CI** (bez `.github/`), žiadny pytest.ini/pre-commit, nič sa nespúšťa automaticky.

## Doplnené — čo v zozname chýba

**A. Komíny.** `engine.prune_small_planes(min_points=400)` komíny len **vyhodí ako šum**; žiadna detekcia/poloha/výška komína. Dataset `rf_psn1f_sdk` triedu Chimney má, nepoužíva sa. Vikiere sú aspoň typizované (`classify_plane_subtype` → "vikier"), komín nie.

**B. Solárne panely.** Nikde v kóde (ani slovo "solar/panel" okrem Roboflow datasetu). Panely na fotke/mračne skresľujú RANSAC fit sklonu aj plochu — treba ich detegovať a vyradiť/označiť.

**C. requirements.txt je chybný.** **Chýba `matplotlib`**, hoci ho importujú `run_v3.py`, `engine.py` (2×), `roof_type.py` — čerstvá inštalácia padne. Naopak `pdal` a `alphashape` sa nikde neimportujú (mŕtve závislosti; pdal sa cez pip na Windows často nedá nainštalovať a blokuje setup). Chýbajú verziové piny (reprodukovateľnosť).

**D. Exporty pre zákazku.** Sú len OBJ/PLY/JSON/SVG/PNG/HTML — **chýba DXF (CAD) a PDF protokol** s rozmermi hrán, čo je výstup, ktorý gesúr/inštalátor reálne potrebuje.

**E. Sieťová odolnosť.** `gis.py` má 3 Overpay mirrory + cache, ale bez retry/backoff; `ortho_fetch`/`geocode` sú single-shot. Chýba batch režim na viac adries naraz a logovanie behov do jedného miesta.

**F. Konfigurácia.** Kľúčové konštanty (EAVE_SNAP_TOL 1,5/2,5 m, GAP_TOL_M, RESIDUAL_WARN_M, min_points) sú natvrdo v kóde — chýba config súbor/UI nastavenia a dokumentácia pre koncového užívateľa (README/changelog, licencie dát ZBGIS/OSM/LLS na jednom mieste).

## TOP 5 priorít (poradie realizácie)

1. **requirements.txt opraviť teraz** (pridať matplotlib, odstrániť pdal/alphashape, piny) — 10 minút, odblokuje čerstvé inštalácie; rovno nadviazať CI workflow (testy + qa_gate) — bod 8.
2. **ZBGIS vektorový obrys** (bod 1) — najlacnejšia cesta k presným hránam; priamo zníži odchýlky 0,92/0,66 m a odstráni FOOTPRINT_GEOMETRY_APPROXIMATED.
3. **Topológia hrán** (body 4+5): snap vrcholov pod 0,05 m, uzavretý odkvapový okruh, eliminácia duplicitných X-hrán, QA kontrola uzavretosti.
4. **Inštančný model do pipeline** (bod 2): overiť/označkovať labely ako inštancie rovín, napojiť roof_mega_v2_s namiesto typu-klasifikátora; k tomu ko-registrácia hrán na ortofoto (bod 3).
5. **GUI v3 zobrazenie** (bod 6) + komíny/solárne panely (A, B) — hotový produkt, ktorý používateľ vidí a ktorý meria správne veci.
