# Výšková analýza — Átriová 16, Trnava

**Dátum:** 2026-08-08
**Adresa:** Átriová 16 (5937/16 + 7960/16M), 917 01 Trnava, Kopánka/Pekné pole
**GPS:** 48.3952363°N, 17.5854374°E

---

## Zistené hodnoty

### 1. Nadmorská výška terénu — PRESNÁ ✅

| Zdroj | Hodnota | Presnosť | Metóda |
|-------|---------|----------|--------|
| **ZBGIS DMR 5.0 (WMS)** | **154.44 m n.m.** | 1 m raster | `GetFeatureInfo` cez `zbgisws.skgeodesy.sk/zbgis_dmr_wms` |
| Open-Meteo API | 155 m n.m. | ~30 m | Interpolácia z globálneho modelu |
| Open-Elevation API | 152 m n.m. | ~30 m | SRTM derivát |
| Copernicus GLO-30 | 153.27 m n.m. | 30 m raster | AWS Copernicus DEM COG |

**Odporúčaná hodnota terénu: 154.44 m n.m.** (ZBGIS DMR 5.0, oficiálny zdroj)

### 2. Výška budovy — NEDOSTUPNÁ BEZ DMP ⚠️

DMP 1.0 (Digitálny model povrchu) **nie je dostupný cez WMS/WCS API**. ZBGIS ho poskytuje iba ako:
- Stiahnutie celých dlaždíc cez **MAPKA** (vyžaduje WebGL2 prehliadač)
- Cloud ZIP balíčky (144-184 GB pre celé Slovensko)

**Copernicus GLO-30 (30 m rozlíšenie)** dáva hrubý odhad ~2.9 m výšky, ale **30 m raster nestačí** na presné meranie jednotlivých budov.

### 3. Typ zástavby (OpenStreetMap)

Okolie Átriovej je zmiešaná obytná zóna:
- Prevažujú rodinné domy (`building=house`)
- Budova 5937/16 je nezaradená (`building=yes`)
- 7960/16M je samostatná budova, tiež nezaradená
- 7751/16A je administratívna budova (`building=office`)
- **Žiadna budova nemá v OSM označený počet podlaží ani výšku**

---

## Čo spraviť pre presnú výšku budovy

### Krok 1: Stiahni DMP 1.0 cez MAPKA

1. Otvor **Google Chrome** (má WebGL2)
2. Choď na https://zbgis.skgeodesy.sk/mapka/
3. V menu vyber **Témy → Terén**
4. Priblíž na Átriovú 16
5. Klikni **Export údajov** → vyber **DMP 1.0**
6. Stiahni príslušný mapový list (ZM 1:5000, dlaždica ~2.5 × 2 km)

### Krok 2: Spusti skript

```bash
pip install rasterio numpy click

# Keď máš DMP_1_0.tif a DMR_5_0.tif:
python tools\lidar_height.py ndsm --dmp DMP_1_0.tif --dmr DMR_5_0.tif -o trnava_ndsm.tif
python tools\lidar_height.py sample --ndsm trnava_ndsm.tif --lat 48.3952363 --lon 17.5854374
```

Výstup bude: **"Výška budovy: X.XX m"** s presnosťou ±0.10 m.

### Alternatíva: CloudCompare + .LAZ

Stiahni .LAZ súbor zo ZBGIS → LLS, otvor v CloudCompare a odmeraj rozdiel Z medzi terénom a strechou.

---

## Technické detaily

- **DMR 5.0 WMS endpoint:** `https://zbgisws.skgeodesy.sk/zbgis_dmr_wms/service.svc/get`
- **DMR vrstva:** Layer `1` = DMR 5.0 (1 m/pixel, EPSG:4326, EPSG:5514, EPSG:8353)
- **DMP 1.0:** Dostupné len cez MAPKA export alebo cloud ZIP (nie cez WMS)
- **DMR 6.0 / DMP 2.0:** Novšia verzia z 2. cyklu LLS (2022-2026), tiež len cez MAPKA
