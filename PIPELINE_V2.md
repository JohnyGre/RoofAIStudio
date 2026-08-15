# RoofAIStudio v2 — Pipeline od nuly (ortofoto + LiDAR + AI tréning)

> Tento dokument je **komplexný špecifikačný prompt** pre nový pipeline.
> Začíname od čistého stavu. Starý kód, staré experimenty a `.cluster/task/*`
> sa NEMAJÚ použiť — všetko sa píše nanovo podľa tohto dokumentu.
> GitHub repo: **názov ostáva `RoofAIStudio`, všetky súbory boli vymazané**,
> ostali len trénované AI moduly.

---

## 0.5 Sériová linka (pipeline) — princíp

Systém funguje ako **sériová linka**: výstup jedného modulu = priamy vstup pre ďalší.
Kľúč k úspechu: správne zladenie **2D vizuálnych dát** (ortofoto) a **3D priestorových dát** (LiDAR).

```
Adresa/GPS → 1. Data Ingestion → 2. 2D Segmentácia → 3. 3D Filtrovanie → 4. Roviny → 5. Topológia → 6. Export
```

| Fáza | Názov | Vstup → Výstup | Knižnica |
|---|---|---|---|
| 1 | **Data Ingestion & Localization** | Adresa → BBox 50×50 m + ortofoto + LAZ | Nominatim, pyproj, PDAL |
| 2 | **2D Semantic Segmentation** | Ortofoto → strešný footprint polygón | YOLO/Detectron2, Shapely |
| 3 | **3D Spatial Filtering** | Maska + LAZ → mračno len strechy | laspy, Open3D |
| 4 | **Plane Fitting** | Mračno → matematické roviny | Open3D/PCL RANSAC |
| 5 | **Topological Reconstruction** | Roviny → vodotesný mesh | Trimesh |
| 6 | **Analytics & Export** | Mesh → metriky + JSON/OBJ/GLTF | Trimesh, numpy |

**Zladenie 2D↔3D:**
- 2D ortofoto dáva **hranice strechy** (footprint, odkvapy) — kde strecha končí.
- 3D LiDAR dáva **výšky, sklony, roviny, hrebene** — geometriu v priestore.
- Footprint z kroku 2 sa premieta ako **vertikálny valec** na 3D mračno (orezanie).

---

## 0.7 Riziká a limity (Hard Truths)

| Riziko | Popis | Riešenie |
|---|---|---|
| **Časový posun (Temporal Misalignment)** | Ortofoto z 2024, LiDAR z 2019 — prestavaná strecha sa nezhoduje | Metadata check: porovnať timestampy zdrojov pred analýzou |
| **LiDAR absorpcia (mŕtve zóny)** | Čierne/lesklé/sklené povrchy neodrážajú laser → diery v mračne | Plocha sa počíta z matematických hraníc polygónu, NIE z počtu bodov |
| **Previsy vegetácie** | Vetva nad strechou klasifikovaná ako trieda 6 (budova) | Surface roughness: strecha = hladká rovina, strom = vysoká variancia normál |
| **Licencovanie AI modelov** | YOLOv8 = AGPL-3.0 → nutnosť zverejniť zdrojový kód | Pre komerčné nasadenie: YOLOX / Mask R-CNN (Apache-2.0/MIT); pre osobný projekt YOLOv8 OK |
| **RAM overload pri LAZ** | Veľké .laz súbory (stovky MB) | PDAL streaming (číta po chunkoch), nie celý súbor naraz |

---

## 0.8 Odporúčaný Open-Source Tech Stack (doplnené o Gemini návrh)

| Modul | Úloha | Preferovaná knižnica |
|---|---|---|
| Data Fetch & Crop | Streamovanie a orez .laz bez RAM overloadu | **PDAL** |
| Coordinate Transform | GPS (WGS84) → S-JTSK, správa projekcií | **pyproj** |
| 2D Segmentation | Detekcia obrysu strechy z ortofotomapy | YOLO / Detectron2 |
| 2D Geometry | Masky, footprinty, boolean operácie | **Shapely** |
| 3D Point Cloud | RANSAC, denoising, normály | **Open3D** alebo PCL |
| 3D Mesh & Topology | Prieniky rovín, triangulácia, export .obj | **Trimesh** |
| LAZ načítanie | .laz → numpy body | laspy + lazrs |

---

## 0. Princíp: Open-source platformy

**Všetko čo sa dá, berieme hotové z open-source platform** — nevynálezame koleso:

| Platforma | Čo odtiaľ použiť | Príklad |
|---|---|---|
| **GitHub** | Modely, šablóny, codebase, dataset linky | ultralytics/ultralytics, pytorch/pytorch |
| **Hugging Face** | Predtrénované modely + datasets | `keremberke/yolov8m-seg` (seg), `timm` backbones |
| **Ultralytics** | YOLO tréning/váhy | `yolov8m.pt`, `yolov8m-seg.pt` |
| **Roboflow / Kaggle** | Datasety striech | roof dataset, aeriálne segmentačné datasety |
| **PyPI / conda** | Knižnice | trimesh, laspy, opencv, torch (CUDA) |
| **three.js** | HTML viewer | lokálna kópia modulov (offline) |

**Pravidlá:**
- Predtrénované váhy sťahovať cez oficiálne API (ultralytics `YOLO("yolov8m.pt")`, HF `snapshot_download`).
- Dataset: použiť existujúci open-source dataset ak vyhovuje, inak vytvoriť vlastný anotovaný.
- CUDA ak je GPU dostupná (`torch.cuda.is_available()`), inak CPU fallback.

---

## 2b. Stiahnutie datasetu + CUDA tréning (nové)

### Dataset

```
data/dataset/
├── images/          # ortofoto výrezy (ZBGIS) 640×640 px
├── labels/          # YOLO .txt anotácie (class x y w h, normalizované)
└── data.yaml        # nc=5, names: [slope_flat, slope_min, slope_poly, slope_trap, slope_tri]
```

**Zdroje datasetu:**
1. **Vlastný** — reťazec: geocode → ZBGIS ortofoto 80 m → YOLO pseudo-labely (starý model) → ručná oprava v GUI.
2. **Open-source** — hľadať na Hugging Face / Kaggle / Roboflow: `roof segmentation aerial`, `building rooftop detection`.
   - HF: `huggingface.co/datasets` (roof/building segmentačné datasety).
   - Roboflow Universe: `roboflow.com/universe` (filtr: aerial, roof).
   - Kaggle: `kaggle.com/datasets` (satellite roof datasets).
3. **Zmixovať** — vlastné ZBGIS ortofoto + open-source satelitné = robustnejší model (ZBGIS má iný vzhľad ako Google Maps).

**Štruktúra YOLO dataset (ultralytics formát):**
```
data.yaml:
  path: data/dataset
  train: images/train
  val: images/val
  nc: 5
  names: ["slope_flat", "slope_min", "slope_poly", "slope_trap", "slope_tri"]
```

### CUDA tréning

```bash
# overiť GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# tréning (v tools/train_yolo.py)
python tools/train_yolo.py --data data/dataset/data.yaml \
    --model yolov8m-seg.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --project ai_models/runs
```

- `device 0` = CUDA GPU, `device cpu` = fallback.
- Výstup: `ai_models/runs/segment/train/weights/best.pt` → skopírovať ako `ai_models/roof_zbgis_best.pt`.
- Pretrénovanie: `--model ai_models/roof_gmaps_v2_last.pt` (transfer learning na ZBGIS ortofoto).

**Metriky po tréningu:**
- mAP50 ≥ 0.6 (cieľ), mAP50-95 ≥ 0.4.
- Confusion matrix → `ai_models/runs/.../confusion_matrix.png`.
- Test: `python tools/test_yolo.py --weights ai_models/roof_zbgis_best.pt --img data/ortho/atriova.jpg`

---

## 2c. Jednoduché GUI (nové)

Hlavné okno = **PySide6**, tri zóny:

```
┌──────────────────────────────────────────────┐
│ Menu: Súbor | Nástroje | Zobraziť | Pomoc    │
├──────────────────────┬───────────────────────┤
│  Ľavý panel (vstupy) │  Pravý panel (výsledky)│
│  - adresa            │  - ortofoto náhľad     │
│  - [GO] tlačidlo     │  - 3D viewer (WebGL)   │
│  - kroky pipeline    │  - tabuľka rovín/hrán  │
│  - log               │  - export tlačidlá     │
└──────────────────────┴───────────────────────┘
```

- **Postupný beh:** tlačidlá spúšťajú kroky 1→12 jeden po druhom, každý má log.
- **3D viewer:** vstavaný QWebEngineView načítavajúci lokálny HTML viewer (three.js, localhost:8080).
- **Všetko v appke** — žiadne externé nástroje.

---

## 1. Cieľ

Z **adresy budovy** automaticky vyrobiť:

1. **Ortofoto** (letecká snímka) — zdroj: **ZBGIS** (primárne) / **OSM** (fallback)
2. **LiDAR mračno bodov (LAZ)** — ZBGIS (bezplatné, CC BY 4.0)
3. **Segmentáciu strechy na roviny** (R1, R2, ...)
4. **Polygóny rovín** s klasifikovanými hranami:
   - `o` = odkvap (okap)
   - `h` = hrebeň
   - `n` = nárožie (uhol od okapu < 90°)
   - `u` = úžľabie (uhol od okapu > 90°)
   - `s` = štít
5. **Výstupy:** JSON (roviny+hrany), OBJ/PLY (farebné), HTML viewer, 2D top-down PNG
6. **Všetko zobrazené v GUI** (RoofAIStudio) — žiadne externé nástroje

---

## 2. Dátové zdroje

| Zdroj | Dáta | URL / API | Licencia |
|---|---|---|---|
| **ZBGIS** | Ortofotomapa (WMS) | `https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get` | CC BY 4.0 |
| **ZBGIS** | LiDAR LAZ mračná | `https://zbgisws.skgeodesy.sk/laz/...` | CC BY 4.0 |
| **ZBGIS** | DMR 5.0 (terén) | WMS GetFeatureInfo | CC BY 4.0 |
| **OpenStreetMap** | Satelitné dlaždice (fallback) | `https://tile.openstreetmap.org/...` | ODbL |
| **Geocoding** | Adresa → GPS | Nominatim (OSM) | ODbL |

**Pravidlá:**
- Primárny zdroj ortofot = **ZBGIS WMS** (rozlíšenie ~10-25 cm/px, aktuálne).
- Ak ZBGIS nedá odpoveď → fallback na OSM dlaždice (zoom 19-20).
- LAZ súbory sa sťahujú podľa GPS buď priamo (ak poznáš mriežku) alebo cez ZBGIS API.
- Všetky stiahnuté dáta sa ukladajú do `data/` (nikdy do `output/` — tam idú len výsledky).

---

## 3. Architektúra (Clean Architecture)

```
RoofAIStudio/
├── app/
│   ├── main.py                 # vstupný bod GUI (PySide6)
│   ├── ui/
│   │   ├── main_window.py      # hlavné okno + menu
│   │   └── panels/             # panely (materiály, doplnky, ...)
│   ├── plugins/                # každý plugin = samostatný modul + tlačidlo
│   │   ├── ortho_plugin.py     # ortofoto: ZBGIS/OSM + Playwright
│   │   ├── lidar_plugin.py     # LiDAR: LAZ → mračno → mesh
│   │   ├── segment_plugin.py   # segmentácia rovín
│   │   ├── edge_plugin.py      # klasifikácia hrán
│   │   ├── export_plugin.py    # JSON/OBJ/PLY/HTML/PNG export
│   │   └── viewer.py           # 3D viewer v GUI
│   ├── core/                   # čistá logika (bez GUI závislostí)
│   │   ├── geocode.py          # adresa → GPS
│   │   ├── ortho_fetch.py      # ZBGIS WMS / OSM dlaždice
│   │   ├── laz_download.py     # LAZ stiahnutie
│   │   ├── pointcloud.py       # LAZ → numpy body, klasifikácia
│   │   ├── segmentation.py     # RANSAC + connected components
│   │   ├── polygons.py         # alpha shape, DP simplifikácia
│   │   ├── edges.py            # klasifikácia hrán
│   │   └── export.py           # výstupy
│   └── ai/
│       └── yolo.py             # YOLO detekcia strechy
├── data/
│   ├── laz/                    # stiahnuté LAZ súbory
│   ├── ortho/                  # ortofotá (jpg/png)
│   └── cache/                  # geocoding cache
├── output/                     # výsledky (JSON, OBJ, PLY, HTML, PNG)
├── ai_models/                  # YOLO modely
└── tools/                      # CLI skripty na testovanie
```

**Pravidlá:**
- `app/core/` = čistá logika, testovateľná bez GUI.
- `app/plugins/` = GUI vrstva, každý plugin pridáva tlačidlo do menu.
- Žiadne externé nástroje (Blender, MeshLab, QGIS) — všetko v appke alebo browser.
- Slovenská diakritika, UTF-8 všade, CSV s `;`.

---

## 4. Pipeline krok za krokom

### KROK 0 — Reset
- Vymazať staré experimenty z `output/` a `.cluster/task/`.
- `git init` nanovo (alebo nový branch `v2`).
- Overiť závislosti: `py -m pip install trimesh numpy scipy shapely alphashape opencv-python pillow laspy lazrs ultralytics playwright PySide6`

### KROK 1 — Adresa → GPS
```
vstup: "Átriová 9309/16, Trnava"
výstup: (lat, lon, display_name)
```
- Nominatim: `https://nominatim.openstreetmap.org/search?q=...&format=json&limit=1`
- Cache do `data/cache/geocode.json` (kľúč = adresa).
- Ak Nominatim vráti viac výsledkov → zobraziť výber v GUI.

### KROK 2 — Ortofoto (ZBGIS primárne, OSM fallback)
**ZBGIS WMS:**
```
GET https://zbgisws.skgeodesy.sk/zbgis_ortofoto_wms/service.svc/get?
  SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap
  &LAYERS=Ortofotomozaika&STYLES=&CRS=EPSG:3857
  &BBOX={xmin,ymin,xmax,ymax}&WIDTH=4096&HEIGHT=4096&FORMAT=image/jpeg
```
- Extent okolo GPS: **80 m × 80 m** (pre YOLO kontext; 40 m je príliš zoomnuté).
- Max zoom = max WIDTH/HEIGHT pri rozumnej veľkosti (4096 px).
- Ak zlyhá → **OSM fallback:** dlaždice `https://tile.openstreetmap.org/{z}/{x}/{y}.png`, zoom 19, poskladať mriežku 3×3 (3840 px).
- Uložiť do `data/ortho/{safe_addr}_ortho.jpg`.

### KROK 3 — YOLO detekcia strechy (voliteľné, pre seed)
- Model: `ai_models/roof_gmaps_v2_last.pt` (trénovaný na satelitoch) — **alebo** nový model trénovaný na ZBGIS ortofotách (lepšia presnosť).
- Detekcia: conf ≥ 0.15, filter na detekcie v strede obrázka.
- Výstup: bounding box strechy → seed pre LiDAR orezanie.
- **Dôležité:** YOLO je len pomocník na seed — **presná geometria ide z LiDARu**.

### KROK 4 — LiDAR (LAZ) z ZBGIS
```
vstup: (lat, lon)
výstup: data/laz/{tile}.laz
```
- Zistiť správny LAZ tile (ZBGIS mriežka, názov podľa súradníc).
- Stiahnuť LAZ (môže byť 20-100 MB).
- Načítať cez `laspy` + `lazrs`:
  - `class 2` = terén
  - `class 6` = budovy
- Orezať na oblasť záujmu (extent 80 m okolo GPS alebo YOLO bbox + margin).

### KROK 5 — Mračno bodov → Mesh (3D)
- Filtrovať outlierov (statistický filter: body ďalej ako 3σ od NN).
- Zjemniť hustotu (voxel downsample na ~5 cm).
- Delaunay triangulácia (2D v X-Y) → mesh.
- Export: `output/{addr}_mesh.obj` (surový, **bez** Laplacian smoothingu).
- Toto je vstup pre segmentáciu rovín.

### KROK 6 — Segmentácia rovín
**RANSAC multi-plane:**
- `distance_threshold = 0.06 m`
- `min_inliers = 200`
- `max_planes = 15`
- Normála: `|nz| > 0.15` (strechovité roviny).

**Connected components split (kľúčová oprava):**
- Po každom RANSAC fite: cKDTree radius graph (radius = 3× medián NN vzdialenosti) + scipy `connected_components`.
- Ak rovina má 2+ signifikantné zhluky (>50 bodov A >5% z celku) → rozdeliť na samostatné roviny.
- **Prečo:** RANSAC lepí dve fyzicky oddelené plochy s rovnakým sklonom (napr. dve rovnaké plochy na opačných stranách budovy).

**Ploché strechy (striešky, terasy):**
- `flat_roof_max_degree = 8.0`
- Roviny so sklonom < 8° → typ `plochá`.
- **Nezahadzovať** density filtrom (ploché plochy majú prirodzene vyššiu density kvôli kolmému dopadu laseru).

### KROK 7 — Polygóny rovín
- **Alpha shape** (nie ConvexHull!): `alphashape.alphashape()`
  - adaptívny `alpha = 3.0 × median(NN distance)`
  - ConvexHull je len fallback (`ALPHA_SHAPE_FAILED` log).
- **DP simplifikácia:** binary-search `cv2.approxPolyDP` → max 8 vrcholov.
- **Validácia:** MultiPolygon filtrovanie (<5% max AND <2% total = noise).
- **Planárnosť:** body roviny ≤ 0.30 m od fitted plane.

### KROK 8 — Overená susednosť + orezanie
- Pre každú dvojicu rovín: vypočítať **presný priesečník** dvoch nekonečných rovín.
- Overiť bodmi: veľa bodov OBOCH mrakov leží blízko priesečnice.
  - `near_tol = 0.50 m`, `min_near_points = 8`, `min_length_m = 2.0`
- Orezanie polygónu polrovinou (Sutherland-Hodgman):
  - **Skúsiť OBE strany, vybrať tú s väčšou plochou** (nie centroid heuristic!).
  - Ak obe strany < 20% plochy → preskočiť (false adjacency).

### KROK 9 — Klasifikácia hrán
- Pre každú hranu: ak leží na priesečnici overených susedov (perp < 0.30 m, alebo < 0.60 m s t-range) → typ podľa **dihedrálneho uhla**:
  - Konvexné (hrebeň/nárožie): obe roviny klesajú od hrany.
  - Konkávne (úžľabie): obe roviny stúpajú k hrane.
- Inak: `okap` (o) alebo `štít` (s).
- **Nárožie vs úžľabie:** uhol od napojeného okapu — <90° nárožie, >90° úžľabie.

### KROK 10 — Snap okapových vrcholov
- Vrcholy z rôznych rovín bližšie ako `eave_snap_tol = 2.5 m` → zlepiť (priemer).
- **Exact priesečníkové vrcholy sa NESNAPujú** (ich geometria je presná).
- Odkvap sa NEROZŠIRUJE — expandujú sa len hrany smerom k hrebeňu a do strán.

### KROK 11 — Export
| Formát | Obsah |
|---|---|
| `output/{addr}_planes.json` | roviny: id, type, area_m2, pitch_deg, vertices, edges (start/end/length/type/exact) |
| `output/{addr}_planes.obj` | farebné roviny + hrany (g = group per plane) |
| `output/{addr}_planes.ply` | vertex colors (RGBA) + edge lines |
| `output/{addr}_pointcloud.ply` | celé mračno, farba podľa roviny / Z-výšky |
| `output/{addr}_viewer.html` | 3D viewer (three.js, lokálne moduly, offline) |
| `output/{addr}_topdown.png` | 2D pohľad zhora (farebné body + obrysy rovín) |

### KROK 12 — Zobrazenie v GUI
- Každý plugin pridá tlačidlo do menu **Tools**.
- Výsledky sa otvárajú:
  - v prehliadači (`http://localhost:8080/...`, server spustený automaticky)
  - alebo priamo v appke (vstavaný viewer / dialóg).
- Žiadny Blender, MeshLab, QGIS.

---

## 5. Čo fungovalo / čo nefungovalo (poučky z v1)

| Zistenie | Dôsledok |
|---|---|
| ConvexHull ničí konkávne tvary (L-shape) | Použiť **alpha shape** |
| RANSAC lepí dve rovnaké plochy | **CC split** po každom fite |
| Centroid heuristic pri orezaní zlyháva | **Obe strany, väčšia plocha** |
| Density filter zabíja ploché strechy | **Výnimka pre sklon < 8°** |
| `file://` blokuje ES moduly | **HTTP server** na localhost:8080 |
| Laplacian smoothing kazí hrany | **Surový OBJ** |
| YOLO na Google Maps nefunguje dobre na ZBGIS | **Pre-trénovať na ortofotách** (v2) |
| LiDAR ±5 cm presnosť Z | **LiDAR je primárny zdroj výšok** |
| Min/max heuristika susednosti = 22 falošných | **Intersection evidence + histogram** |
| `side_val` centroid = zlé strany pri R3 | **Both-side clipping** |
| 40 m extent = príliš zoomnuté pre YOLO | **80 m extent** |

---

## 6. Overovanie (QA)

Pre každú adresu po behu pipeline skontrolovať:

1. **Počet rovín:** 4-12 (podľa typu strechy), žiadna `LOW_CONFIDENCE` okrem naozaj riedkych.
2. **Plochy:** súčet plôch ~ očakávaná plocha strechy (z ortofota).
3. **Žiadne medzery:** susedné polygóny zdieľajú presne rovnakú hranu.
4. **Hrany:** každá hrana má typ (o/h/n/u/s), žiadna neklasifikovaná.
5. **Plochá strecha:** sklon < 8° = typ `plochá`.
6. **Vizuálna kontrola:** 3D viewer + top-down PNG — polygóny sedia na ortofoto.

Automatické kontroly v `tools/qa.py`:
- `--check-gaps` (medzery medzi plochami)
- `--check-edges` (typová distribúcia)
- `--check-areas` (súčet vs očakávanie)
- `--check-flat` (ploché strechy detegované)

---

## 7. Playwright (pre prípad, že WMS nestačí)

ZBGIS WMS je primárna cesta (jednoduchá HTTP GET). **Playwright sa používa len keď:**
- ZBGIS dáva 403/429 (anti-scraping),
- treba interaktívnu MAPKA aplikáciu (výber bodu, stiahnutie ZIP),
- OSM dlaždice nefungujú.

Playwright flow:
```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto("https://zbgis.skgeodesy.sk/", wait_until="networkidle")
    # ... klik na vyhladavanie, zadanie adresy, vyber, screenshot ortofota ...
    page.screenshot(path="data/ortho/{addr}_playwright.png")
    browser.close()
```

---

## 8. Testovacia adresa

**Átriová 9309/16, Trnava** (predošlé dáta):
- LAZ tile: `_triov__9309_16h__917_01_trnav` (extent ~40×39 m)
- LiDAR Z rozsah: 0.8 – 9.0 m
- Očakávané: 8-10 rovín (sedlová s átriom + plochá strieška)

---

## 10. Navrhovaný postup (prvý beh, poradie prác)

### Fáza A — Príprava (30 min)
1. Vytvoriť adresárovú štruktúru (kapitola 3) v `RoofAIStudio/`.
2. `py -m venv .venv` + `requirements.txt` (trimesh, numpy, scipy, shapely, alphashape, opencv, pillow, laspy, lazrs, ultralytics, playwright, PySide6, **pyproj**, **open3d**, **pdal**).
3. Skopírovať predtrénovaný model `ai_models/roof_gmaps_v2_last.pt` ako baseline.

### Fáza B — Kroky 1-2 (Data Ingestion) ✅ prvý míľnik
4. `app/core/geocode.py` — adresa → GPS (Nominatim + cache).
5. `app/core/ortho_fetch.py` — ZBGIS WMS ortofoto 80×80 m (4096 px) + OSM fallback.
6. `app/core/transform.py` — GPS (WGS84) → S-JTSK (pyproj) pre LAZ vyhľadávanie.
7. **Test:** Átriová 9309/16 → GPS + ortofoto uložené v `data/ortho/`. ✅

### Fáza C — Krok 4 (LiDAR)
8. `app/core/laz_download.py` — nájsť + stiahnuť LAZ tile podľa S-JTSK súradníc.
9. `app/core/pointcloud.py` — laspy načítanie, class filter (2=terén, 6=budova), voxel downsample, SOR denoising.
10. **Test:** mračno budovy z ÁtrioVEJ v `data/laz/`. ✅

### Fáza D — Kroky 3+4 (2D segmentácia + roviny)
11. `app/ai/yolo.py` — YOLO segmentácia ortofota → maska + footprint (Shapely polygón).
12. `app/core/segmentation.py` — RANSAC + CC split + ploché strechy (<8°).
13. **Test:** 8-10 rovín pre Átriovú. ✅

### Fáza E — Kroky 5-6 (Topológia + Export)
14. `app/core/polygons.py` — alpha shape, DP simplifikácia, both-side clipping.
15. `app/core/edges.py` — klasifikácia hrán (dihedrálny uhol).
16. `app/core/export.py` — JSON, OBJ, PLY, HTML viewer, top-down PNG.
17. **Test:** kompletný výstup pre Átriovú. ✅

### Fáza F — GUI + QA
18. `app/ui/main_window.py` — PySide6 okno, tlačidlá pre kroky, 3D viewer.
19. `tools/qa.py` — automatické kontroly (gaps, edges, areas, flat).
20. **Test:** celý pipeline cez GUI. ✅

---

## 11. Čo NIE robiť (antivzory z v1)
- Nehromadiť experimenty v `.cluster/task/` — každý modul má svoje miesto.
- Nepoužívať ConvexHull na tvary — alpha shape.
- Nespúšťať Blender/MeshLab/QGIS — všetko v appke/browseri.
- Neprepisovať fungujúci kód naslepo — najprv test, potom zmena.
- Neukladať heslá do kódu — `os.getenv()`.
- Nevyhadzovať ploché strechy density filtrom.

---

## 9. Prvý beh (checklist)

- [ ] `git init` + nový branch `v2`
- [ ] Vytvoriť adresárovú štruktúru (kapitola 3)
- [ ] Stiahnuť predtrénovaný model (ultralytics/HF)
- [ ] Dataset: open-source alebo vlastný (ZBGIS) → `data/dataset/`
- [ ] CUDA tréning → `ai_models/roof_zbgis_best.pt`
- [ ] `tools/qa.py` kostra
- [ ] Krok 1: geocode Átriová → GPS ✅
- [ ] Krok 2: ZBGIS ortofoto 80 m ✅ (fallback OSM)
- [ ] Krok 4: LAZ stiahnutie + načítanie ✅
- [ ] Krok 5-6: mesh + RANSAC + CC split ✅
- [ ] Krok 7-8: alpha shape + orezanie ✅
- [ ] Krok 9-10: hrany + snap ✅
- [ ] Krok 11-12: export + GUI ✅
