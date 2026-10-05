# RoofAIStudio v3 — čítaj ma (pre agentov a vývojárov)

Táto vetva obsahuje **v3 pipeline** na modelovanie striech: z adresy/GPS vyrobí roviny strechy,
klasifikované hrany (odkvap/hrebeň/nárožie/úžľabie/štít), presné polygóny v S-JTSK a balík
výstupov (JSON kontrakt, DXF, HTML report, izometrický výkres, GeoTIFF, GeoJSON).

> Stav: **overené**. QA: 0 chýb / 0 varovaní; regresia na 3 adresách: PASS; beh deterministický
> (dva po sebe idúce behy dávajú identické výsledky). Príklad: Trnava 7751/16A → 11 rovín,
> 328,9 m² vs pôdorys 343,7 m² (pomer 0,96).

---

## 1. Architektúra (kto je autorita pre čo)

| Zdroj | Rola | Poznámka |
|---|---|---|
| **LiDAR (LAZ, trieda 6)** | **autorita geometrie** — roviny, sklony, výšky | LAZ zo ZBGIS/MAPKA (S-JTSK JTSK03 + Bpv) |
| **GIS obrys budovy** | plánometrická autorita (orezanie mračna) | OSM (Overpass) s geometriou; ZBGIS WFS pripravený (`gis.zbgis_footprint`) |
| **Vycvičený model (YOLO-seg)** | typológia + planimetrický podklad + QA | `ai_models/roof_gmaps_v2_last.pt` (5 tried typov striech) |
| **Ortofoto ZBGIS (WMS)** | podklad pre vizuálnu kontrolu a prekrytie | 4096 px, ≈0,7 cm/px |
| **LLM** | **zámerne nie je** v geometrii | len plán/QA/reportovanie (mimo tohto kódu) |

Princíp: **hrany z obrazu/mračna, roviny z mračna, rozhoduje deterministický engine.**

## 2. Spustenie

```powershell
# 1) prostredie (raz)
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2) celý v3 beh na adrese (GPS súradnice alebo adresa)
.venv\Scripts\python.exe tools\run_v3.py --lat 48.3955928 --lon 17.5856480 --name Triova --ortho

# varianty
tools\run_v3.py --address "Átriová 9309/16, Trnava" --ortho --preprocess
tools\run_v3.py --lat ... --lon ... --name X            # bez ortofota (rýchle)
tools\ci.cmd                                            # celý CI: testy → selfcheck → QA → regresia → export
.venv\Scripts\python.exe tools\regression.py            # regresia na 3 adresách
.venv\Scripts\python.exe tools\qa_gate.py <kontrakt.json>
.venv\Scripts\python.exe tools\v3_export.py <kontrakt.json>   # DXF + HTML report
```

## 3. Mapa súborov

```
app/core/
  contract.py      # verzovaný dátový kontrakt (schema 3.0.0) + validácia
  engine.py        # geometrický engine: segmentácia, roviny, hrany, komíny, topológia
  fusion.py        # deterministická fúzia zdrojov (priority, konflikty, flagy)
  edges.py         # most desktop -> kontrakt: klasifikované hrany h/n/u z priesečníc
  gis.py           # obrys budovy (OSM + pripravený ZBGIS), cache-first
  ortho.py         # ZBGIS WMS snímka + prekrytie geometrie
  preprocess.py    # GeoTIFF + dlaždicové masky + GeoJSON (zrozumiteľný formát)
  qa.py            # kontroly geometrie (trieda/sklon, plochy, medzery, low_conf)
  registration.py  # CRS normalizácia, rezíduá, časový nesúlad zdrojov
  vision.py        # vycvičený segmentačný model → obrys/masky (S-JTSK)
app/plugins/
  contract_exporter.py   # export kontraktu v3 + QA report
tools/
  run_v3.py        # HLAVNÝ orchestrátor (CLI)
  regression.py    # regresný test na 3 adresách (kritériá kvality)
  render_iso.py    # izometrický technický výkres (SVG + PNG)
  v3_export.py     # DXF (vrstvy) + HTML report
  qa_gate.py       # QA brána s návratovým kódom
  core_selfcheck.py# overenie jadra na reálnych dátach
  ci.cmd           # CI reťaz
tests/
  test_core_contract.py   # 13 kontrol (kontrakt, QA, fúzia)
```

## 4. Dátový tok

1. **GPS / adresa** → súradnice (WGS84) → S-JTSK/JTSK03 (EPSG:8353).
2. **GIS obrys** (OSM/ZBGIS) → orezanie mračna na jednu budovu.
3. **LiDAR** (LAZ trieda 6) → výškový pás ±7 m → voxel 0,15 m → oddelenie budovy.
4. **Vision model** → obrys + masky typov; co-registrácia s mračnom (len ak IoU ≥ 0,45).
5. **Engine**: RANSAC → CC split → zlúčenie koplanárnych → pravidelný tvar plochy →
   orezanie susedmi → snap vrcholov (spoločné hrany) → klasifikácia hrán (o/h/n/u/s) →
   komíny a panely → uzavretý obvod odkvapu.
6. **Kontrakt v3** (JSON) + **QA report** + registrácia/časové flagy.
7. **Výstupy**: izometrický výkres (SVG/PNG), kontrolný obrázok na ortofote, top-down PNG,
   DXF po vrstvách, HTML report, (voliteľne) GeoTIFF + GeoJSON masky.

## 5. Kontrakt (schema 3.0.0) — čo v ňom je

```json
{
  "schema_version": "3.0.0",
  "crs": "EPSG:8353",
  "address": "...", "gps": {"lat": ..., "lon": ...},
  "ground_mnm": 154.5, "footprint_m2": 343.69, "roof_area_m2": 328.89,
  "sources": [{"id": "lidar_lls", "role": "lidar", "crs": "EPSG:8353", "acquired_year": 2018}, ...],
  "planes": [{
      "id": "R1", "type": "sedlová/valbová", "pitch_deg": 30.5, "area_m2": 54.3,
      "azimuth_deg": 48.6, "rmse_m": 0.024,
      "vertices": [[x, y, z], ...],
      "edges": [{"id": "h1", "type": "h", "length_m": 8.2, "start": [..], "end": [..], "exact": true}]
  }]
}
```

Typy hrán: `o` odkvap · `h` hrebeň · `n` nárožie · `u` úžľabie · `s` štít.
`exact: true` = hrana je presná priesečnica rovín (nie odhad z obrysu bodov).

## 6. Overenie (čo má agent spustiť po zmenách)

```powershell
tools\ci.cmd
# očakávané: "CI OK" + v regresnom reporte PASS na 3 adresách
# kritériá: 0 chýb kontraktu · QA bez chýb · ≥3 roviny · ≥3 klasifikované hrany · pomer 0,55–1,8
```

Vzorky výstupov (malé, čitateľné): `docs/v3/priklady/` — kontrakt JSON, QA JSON, regresný report,
izometrický výkres (SVG), masky (GeoJSON), preprocess meta.

## 7. Známe medzery a ďalšie kroky

1. **ZBGIS vektorový obrys budovy** — verejné WFS vracia 404; doplniť konkrétnu vrstvu
   (`gis.zbgis_footprint`) alebo nahrať export z MAPKA (SHP/GeoJSON/DXF).
2. **Inštančný segmentačný model** — súčasný model delí *typy* striech, nie jednotlivé roviny.
   Plán: vygenerovať tréningové dáta z overených v3 výstupov a dotrénovať (ultralytics, CUDA).
3. **Novší LiDAR** (2./3. cyklus, 15–45 b/m²) — presnejšie detaily (vikiere, komíny).
4. Odchýlka hrán voči ortofotu je meraná 0,66–0,92 m (dominuje fragmentácia plôch).

## 8. Licencie a zdroje dát

- ZBGIS ortofoto a mračno bodov: **CC BY 4.0** (GKÚ Bratislava) — pri publikácii uviesť zdroj.
- OpenStreetMap obrysy: **ODbL**.
- Beží lokálne; žiadne dáta sa neposielajú do cloudu (okrem voliteľného vizuálneho posúdenia).

