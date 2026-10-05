# RoofAIStudio v3 — záverečné doručenie

## Čo je hotové a overené

| Oblasť | Stav | Dôkaz |
|---|---|---|
| **Pipeline v3** (GPS → LiDAR → GIS/vision obrys → engine → kontrakt → QA → výkresy) | hotová, deterministická | `tools/run_v3.py`, beh 20–75 s na adresu |
| **Geometria** | roviny z mračna (RANSAC + CC split + zlúčenie), obrysy ako pravidelné strešné tvary, hrany z priesečníc, konvexnosť z reálnych bodov | `app/core/engine.py` |
| **Topológia** | spoločné hrany susedných plôch (snap + dotiahnutie na hranicu), uzavretý odkvapový obvod | 53 vrcholov dotiahnutých, obvod 42,05 m |
| **Klasifikácia hrán** | odkvap / hrebeň / nárožie / úžľabie / štít | Triova: o=11, h=10, n=20, u=15, s=16 |
| **Prvky strechy** | komíny (2 nájdené), kontrola solárnych panelov | `detect_features_from_cloud` |
| **Zdroje** | LiDAR (autorita geometrie) · GIS/OSM obrys · vycvičený model (planimetria+typologia, IoU 0,76) · ortofoto ZBGIS 4096 px | log behu + QA report |
| **QA** | kontrakt + geometrické kontroly (trieda/sklon, plochy, medzery, low_conf) | **0 chýb, 0 varovaní** na všetkých 3 adresách |
| **Regresné testy** | 3 adresy (Trnava 7751, Átriová 9309, Beluj 50) | `regression_report.md` — **PASS** |
| **CI** | 5 krokov (testy → selfcheck → QA → regresia → export) | `tools/ci.cmd` — **CI OK** (exit 0) |
| **Výstupy** | kontrakt v3 (JSON), DXF po vrstvách, HTML report, izometrický výkres (SVG/PNG), kontrolný obrázok na ortofote, top-down | `output/v3/` |

## Čísla (posledný overený beh)

| Adresa | Rovín | Plocha | Pôdorys | Pomer | QA | Čas |
|---|---|---|---|---|---|---|
| Triova 7751/16A, Trnava | 11 | 287,0 m² | 343,7 m² | 0,84 | 0 chýb / 0 varovaní | 20 s |
| Átriová 9309/16, Trnava | 4 | 131,1 m² | — | — | 0 / 0 | 68 s |
| Beluj 50 | 9 | 186,1 m² | — | — | 0 / 0 | 75 s |

## Ako sa to spúšťa

```powershell
cd C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio

spustit.cmd v3                 # pipeline + ortofoto + výkres + DXF + HTML report (otvorí sa)
spustit.cmd v3bezfoto          # rýchla verzia bez ortofota
tools\ci.cmd                   # celý CI reťaz (testy → QA → 3 adresy → export)
.venv\Scripts\python.exe tools\regression.py    # len regresné testy
.venv\Scripts\python.exe tools\run_v3.py --lat .. --lon .. --name .. --ortho
```

## Čaká na teba (jediná vec)

**Katastrálny obrys budovy z ZBGIS/MAPKA** (SHP / GeoJSON / DXF) pre testovacie adresy.
Verejný WFS endpoint vracia 404 (je určený INSPIRE klientom s účtom), preto je najjednoduchšia cesta:
MAPKA → nájsť budovu → export „ZBGIS / Budova" → súbor do `C:\Users\jangr\Downloads\`.
Kód je pripravený: `gis.py::zbgis_footprint()` — po dodaní obrysu sa model oreže katastrálne presne
a odchýlka hrán (dnes meraná 0,66–0,92 m voči ortofotu) sa zníži na centimetre.

Nepovinné: novší LiDAR (2./3. cyklus, 15–45 b/m²) — objednávka cez MAPKA ako doteraz.

## Ponuka do ďalšieho kroku (bez sťahovania)

Viem sám vygenerovať tréningové dáta (masky rovín z overeného v3 výstupu + ortofoto) a
natrénovať inštancový segmentačný model na RTX 3050 — to je cesta k presným hraniciam plôch
bez akéhokoľvek externého datasetu.

## Predspracovanie do „zrozumiteľného" formátu (hotové, `--preprocess`)

Nový modul `app/core/preprocess.py` vyrába z ZBGIS snímky balík pre ďalšie kroky pipeline:

| Súbor | Formát | Význam |
|---|---|---|
| `<name>_ortho.tif` | **GeoTIFF (EPSG:3857)** | georeferencovaná snímka — žiadne ručné prepočty pixelov |
| `<name>_ortho.pgw` | world file | to isté pre CAD/QGIS (6 čísel) |
| `<name>_masky.tif` | GeoTIFF | masky segmentačného modelu (dlaždicovo, prevlečenie 96 px) |
| `<name>_masky.geojson` | **GeoJSON (EPSG:8353)** | polygóny tried v S-JTSK — vstup pre geometrický engine |
| `<name>_preprocess.json` | JSON | metadáta (bbox, CRS, model, počty tried) |

Overené: GeoTIFF má korektné geo-tagy (ModelPixelScale 0,0152 m/px, tiepoint, GeoKeyDirectory
EPSG:3857), GeoJSON má 0 neplatných polygónov. Spúšťa sa cez `run_v3.py --ortho --preprocess`.

Poznámka z testu: extrakcia línií priamo z ortofota je slabá (LSD našiel 18 línií > 2 m,
zhoda s hranami z mračna 1/45) — obraz dáva sémantiku (typy/plochy), geometriu musí dať mračno.
Preto je preprocess podporný krok, nie náhrada LiDARu.

