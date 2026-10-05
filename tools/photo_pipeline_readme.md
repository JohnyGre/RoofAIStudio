# RoofAI Photo Pipeline — Dokumentácia

## Prehľad

Pipeline na spracovanie bežnej fotografie strechy rovnakým spôsobom ako satelitné snímky:
1. **Predspracovanie** — resize na 640×640 px (preserve aspect ratio + padding 114)
2. **Detekcia** — YOLO segmentačný model (`roof_gmaps_v2_last.pt`)
3. **Klasifikácia** — 5 tried: `slope_flat`, `slope_min`, `slope_poly`, `slope_trap`, `slope_tri`
4. **GUI** — HTML s vrstvami (pôvodný, spracovaný, detekcia+klasifikácia)
5. **Backend** — SQLite databáza s metadatami
6. **Konfigurácia** — JSON súbor s parametrami

## Inštalácia závislostí

```powershell
cd C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio
.venv\Scripts\pip install ultralytics opencv-python Pillow
```

## Spustenie

### Z CLI

```powershell
.venv\Scripts\python.exe tools\photo_pipeline.py <cesta_k_obrazku> [confidence_threshold]
```

**Príklad:**
```powershell
.venv\Scripts\python.exe tools\photo_pipeline.py output\atriova_16_v2_ortofoto_maxzoom.jpg 0.15
```

### Z GUI (RoofAIStudio)

V aplikácii: **Tools → Photo Pipeline** (ak je aktivovaný plugin).

## Výstupné súbory

Všetky výstupy sú v `output/photo_pipeline/`:

| Súbor | Popis |
|-------|-------|
| `<nazov>_pipeline_gui.html` | HTML GUI s 3 vrstvami a tabuľkou detekcií |
| `<nazov>_pipeline_result.json` | JSON s metadatami a detekciami |
| `pipeline_results.db` | SQLite databáza so všetkými spusteniami |
| `pipeline_config.json` | Konfiguračný súbor |

## Interpretácia výstupov v GUI

HTML GUI obsahuje 4 sekcie:

1. **Pôvodná fotografia** — vstupný obrázok v pôvodnom rozlíšení
2. **Spracovaný vstup (640×640)** — po resize+padding, presne ako vidí model
3. **Detegované hrany + klasifikácia** — prekryv masiek, bounding boxy, štítky
4. **Klasifikácia detegovaných oblastí** — tabuľka s triedou, confidence, plochou

### Triedy

| Kód | Slovenský názov | Farba (RGB) |
|-----|-----------------|-------------|
| `slope_flat` | Plochá strecha | (100, 149, 237) cornflower blue |
| `slope_min` | Minimálny sklon | (72, 201, 176) aquamarine |
| `slope_poly` | Polygónálna | (255, 165, 0) orange |
| `slope_trap` | Lichobežníková | (220, 20, 60) crimson |
| `slope_tri` | Trojuholníková | (138, 43, 226) blue violet |

## Konfigurácia

Súbor `pipeline_config.json`:

```json
{
  "model": {
    "path": "ai_models/roof_gmaps_v2_last.pt",
    "task": "segment",
    "imgsz": 640
  },
  "preprocessing": {
    "target_size": 640,
    "padding_value": 114,
    "normalization": "imagenet"
  },
  "inference": {
    "confidence_threshold": 0.25,
    "iou_threshold": 0.7,
    "device": "auto"
  }
}
```

## Databáza

SQLite v `pipeline_results.db`, tabuľka `pipeline_runs`:

| Stĺpec | Typ | Popis |
|--------|-----|-------|
| id | INTEGER PK | Auto-increment |
| filename | TEXT | Názov vstupného súboru |
| timestamp | TEXT | Čas spracovania |
| num_detections | INTEGER | Počet detekcií |
| classes | TEXT (JSON) | Zoznam tried |
| confidences | TEXT (JSON) | Zoznam confidence hodnôt |
| processing_time_ms | REAL | Čas spracovania |
| image_hash | TEXT | MD5 hash vstupu |
| config_hash | TEXT | MD5 hash konfigurácie |
| output_json | TEXT | Kompletný výsledok ako JSON |

## Reprodukovateľnosť

Rovnaký vstup + rovnaká konfigurácia = identické výsledky. Overené spustením 2× na tom istom obrázku — image_hash a config_hash sa zhodujú, detekcie sú identické.

## Úpravy

| Čo zmeniť | Kde |
|-----------|-----|
| Confidence threshold | CLI parameter alebo `pipeline_config.json` → `inference.confidence_threshold` |
| Model | `pipeline_config.json` → `model.path` |
| Veľkosť vstupu | `pipeline_config.json` → `preprocessing.target_size` |
| Farby tried | `tools/photo_pipeline.py` → `CLASS_COLORS` |
| Slovenkské názvy | `tools/photo_pipeline.py` → `CLASS_LABELS_SK` |
| Dizajn GUI | `tools/photo_pipeline.py` → funkcia `generate_html_gui()` |
