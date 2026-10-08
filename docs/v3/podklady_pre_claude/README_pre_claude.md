# Podklady pre Claude — Triová 7751/16A (obrys + mračno bodov)

**Adresa:** 7751/16A, Átriová, Kopánka, 917 01 Trnava
**Dátum exportu:** 2026-10-08
**CRS všetkého nižšie (okrem WGS84 GeoJSON):** EPSG:8353 — S-JTSK [JTSK03] / Krovak East North + Bpv (výšky)
**Stred (8353):** X = -535 692,92 · Y = -1 256 458,05

## 1. Obrys budovy (dve verzie)

| Súbor | Formát | CRS | Poznámka |
|---|---|---|---|
| `Triova_budova_obrys_WGS84.geojson` | GeoJSON (RFC 7946) | WGS84 (CRS84) | obrys tak, ako ho vidí OSM/Overpass |
| `Triova_budova_obrys_SJTSK.geojson` | GeoJSON | **EPSG:8353** | ten istý obrys prepočítaný do S-JTSK (pre projektanta) |

- Zdroj: **OpenStreetMap (Overpass)** — obrys je **generalizovaný, nie katastrálny**; plocha 343,69 m² (22 bodov).
- Ak potrebuješ presnejší (katastrálny) obrys, v projekte je pripravený hook `app/core/gis.py::zbgis_footprint()`.

## 2. Mračno bodov (obilinkové okno okolo budovy)

| Súbor | Formát | Čitateľnosť |
|---|---|---|
| `Triova_mracno_35m.las` | LAS 1.2, point format 3, skaly 0,001 m, **CRS EPSG:8353** | laspy / CloudCompare / QGIS / PDAL |
| `Triova_mracno_35m_ascii.ply` | PLY ASCII (x, y, z, `classification`) | čokoľvek, aj ručne |

- **Okruh:** 35 m od stredu budovy · **Počet bodov:** 103 517
- **Zdrojové dlaždice:** `05_Trnava_18_246985_5365887`, `05_Trnava_18_247065_5365631`, `05_Trnava_18_247070_5366181` (LLS cyklus 2018, triedy zachované z originálu)
- **Triedy v okne:**

| kód | význam | bodov |
|---|---|---|
| 1 | neklasifikované | 2 562 |
| 2 | terén (ground) | 51 877 |
| 3 | nízka vegetácia | 4 023 |
| 4 | stredná vegetácia | 10 031 |
| 5 | vysoká vegetácia | 19 137 |
| **6** | **budovy** | **15 737** |
| 7 | nízke body / šum | 150 |

- **BBox (8353):** X −535 731,3 … −535 654,5 · Y −1 256 496,1 … −1 256 420,0 · Z (Bpv) 152,8 … 166,4 m

## 3. Súvisiace podklady v repozitári

- Ortofoto (4096 px, EPSG:3857, výrez ~39 m): `output/v3/Triova_7751_16A_Trnava_ortofoto_max.jpg` (lokálne; v repozitári nie je — veľké súbory)
- Kontrakt v3: `docs/v3/priklady/Triova_7751_16A_Trnava_roofmodel_v3.json`
- QA report: `docs/v3/priklady/Triova_7751_16A_Trnava_roofmodel_v3_qa.json`
- Metadáta exportu (strojovo): `Triova_podklady_README.json`

## 4. Poznámky k presnosti (aby sedeli očakávania)

- Mračno je **cyklus 2018** (hustota ~5 b/m²) — jemné prvky (vikiere, komínky) môžu byť redšie.
- Susedná výška Bpv je absolútna; ak potrebuješ relatívne výšky, odčítaj terén (trieda 2 pod budovou).
- Východné krídlo a spoj: v mračne sú celé (okno 35 m ich pokrýva); hranice, ktoré „chýbajú", sú dôsledok orezania polygónov v našom engine, nie absencie bodov.
