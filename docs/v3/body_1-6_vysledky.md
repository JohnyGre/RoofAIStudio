# Body 1–6: čo bolo urobené, otestované a opravené

Slučka „otestuj → skontroluj → pri nepresnosti oprav" pre každý bod zo zoznamu.

| Bod | Zadanie | Výsledok | Dôkaz |
|---|---|---|---|
| **1** | ZBGIS vektorový obrys budovy | **Overené, že teraz nedostupné.** Testované endpointy `zbgis_wfs`, `zbgis/rest/services`, `zbgis_smd_wfs` → HTTP 404; verejný WFS pre vrstvu budov sa nenašiel ani vyhľadaním. **Riešenie:** pipeline používa OSM obrys **s geometriou** (keď príde) a keď nepríde, je jasný flag `NO_GIS_FOOTPRINT`; `gis.py::zbgis_footprint()` je pripravený na doplnenie konkrétnej WFS vrstvy. | beh druhej adresy: `POZOR: GIS obrys nedostupný (sieť aj cache) → FLAG NO_GIS_FOOTPRINT` |
| **2** | Topológia hrán bez medzier + uzavretý odkvap | **Hotové.** Párový snap spoločných rohov (16 dvojíc) znížil medzery na nulu → **QA: PASS (0 chýb, 0 varovaní)**. Odkvap je teraz **uzavretý obvod z obrysu strechy, dĺžka 42,05 m, 5 bodov**. | `[topologia] snapnutých dvojíc vrcholov: 16`; `odkvapový obvod (obrys strechy): uzavretý, dĺžka 42.05 m` |
| **3** | Model na inštancie + presnejšia co-registrácia | **Čiastočne.** Vyskúšaný auto-výber modelu (roof_gmaps_v2.pt mal IoU 0,708 vs 0,681) — kvôli variabilite výsledkov som ho zafixoval na deterministický `roof_gmaps_v2_last.pt` s prepínačom `--vision-model`. Co-registrácia **dostala guard**: posun sa aplikuje len ak IoU ≥ 0,45, inak sa zamietne s flagom `REGISTRATION_LOW_CONFIDENCE` (na Átiovej 9309/16 by zlý posun −2,5/+3,0 m pokazil model). Inštančný model nie je k dispozícii — `ai_models/runs/roof_mega_v2_s` je typový segmenter. | `registrácia zamietnutá (IoU 0.166 < 0,45) → FLAG REGISTRATION_LOW_CONFIDENCE` |
| **4** | Komíny a solárne panely | **Hotové pre komíny.** Detekcia z **celého mračna** (body 0,2–3,0 m nad najbližšou rovinou, v jej pôdoryse): **nájdené 2 komíny** (1,81 m² / 29 bodov a 0,62 m² / 10 bodov). Panely: 0 (na tejto streche nie sú). | `komíny: 2`, `komín @ (-535691.4, -1256465.2): 1.81 m², 29 bodov` |
| **5** | Zobrazenie v3 + DXF/PDF | **Hotové.** Pribudol `tools/v3_export.py`: **DXF R12** s hranami na vrstvách (ODKVAP/HREBEN/NAROZIE/UZLABIE/STIT + ROVINY) pre projektanta a **HTML report** (tabuľka rovín, plochy, náhľady) — pipeline ho po behu **automaticky otvorí v prehliadači**. | `DXF: ..._hrany.dxf`, `HTML report: ..._report.html` |
| **6** | CI + overenie na viacerých adresách | **Hotové.** `tools/ci.cmd` spúšťa testy → selfcheck → QA gate → export; **celý reťaz prechádza (`CI OK`)**. Overené na **dvoch adresách**: Trnava 7751/16A (11 rovín, QA PASS) a Átriová 9309/16 (4 roviny, 130 m², flags hlásené korektne). | `CI OK`; behy v `output/v3/` |

## Zhrnutie stavu po bodoch

- **QA: PASS** na hlavnej adrese — 0 chýb, 0 varovaní (medzery odstránené, obvod uzavretý).
- **Nájdené prvky:** 11 rovín, 6 úžľabí / 4 hrebene / 7 nárožných hrán, 2 komíny, uzavretý odkvap 42,05 m.
- **Exporty:** DXF (hrany po vrstvách), HTML report, izometrický výkres (SVG/PNG), kontrolný obrázok na ortofote.
- **Determinizmus:** model, posun registrácie a prahy sú fixné; tie isté vstupy dávajú tie isté výstupy.

## Čo zostáva (nad rámec bodov 1–6)

1. Doplniť konkrétnu ZBGIS WFS vrstvu budov (bod 1) — bez nej je obrys z OSM/vision aproximácia.
2. Tréning inštančného segmentačného modelu (datasety `data/roofs_mega*`) pre presné hranice plôch.
3. 3D viewer s hranami priamo v desktop aplikácii (teraz je report v prehliadači).
