# Iteratívne ladenie hrán (6 kôl: prepíš → spusť → vizuálne skontroluj)

Cieľ: hrany strechy na výkrese presne podľa reality, overované prekreslením na ortofoto.

| Kolo | Čo som zmenil | Nález kontroly | Výsledok |
|---|---|---|---|
| 1 | východisko (hrany z obrysu bodov) | odkvapy mimo obrysu aj vnútri strechy; „hrebeň" bol v skutočnosti úžľabie; úžľabia chýbali | nevyhovuje |
| 2 | presné hrany z priesečníc rovín; konvexnosť podľa maxima výšky | hrebene a nárožia zmizli, úžľabia ako umelé obdĺžniky | nevyhovuje |
| 3 | odkvapy z obrysu celej strechy; zapojenie vycvičeného modelu (obrys) | odkvapy stále naprieč strechou | nevyhovuje |
| 4 | co-registrácia modelu a mračna (IoU 0,577 → **0,756**) | úžľabia **už sedia** na skutočných líniách; hrebene a nárožia chýbajú | čiastočne |
| 5 | konvexnosť z **reálnych bodov mračna** (nie z extrapolácie rovín) | odkvapy dobré (malé odchýlky), hrebeň dobrý, nárožia dobré s malými chybami, úžľabia menej presné | použiteľné |
| 6 | skrátenie úžľabí na úsek s obojstrannými bodmi | odkvapy prevažne sedia, hrebeň čiastočne, nárožia/úžľabia posunuté | čiastočné chyby |

## Kľúčové opravy v kóde (v `app/core/engine.py`)

1. **Konvexnosť z reálnych bodov** — `_fold_convex` berie body z pásu 0,3–1,5 m okolo
   priesečnice a porovnáva Δz s výškou línie; medián < 0 → hrebeň/nárožie, > 0 → úžľabie.
   Predtým sa porovnávala extrapolácia rovín, čo dávalo systematicky nesprávny výsledok
   (raz „všetko hrebeň", raz „všetko úžľabie").
2. **Smer spádu** — `_down_dir` opravený (spád ide v smere `(n_x, n_y)`).
3. **Odkvap z vonkajšieho obvodu** — hrana, za ktorou už nie sú body žiadnej roviny
   („min distance > 0,9 m") → `o`. Nahradilo pôvodné „najnižšia hrana", ktoré dávalo
   odkvapy aj vnútri strechy.
4. **Snapping vrcholov na priesečnice** (`snap_vertices_to_intersections`) — hranice plôch
   sa pritlačia na presnú priesečnicu, takže hrana sa dá klasifikovať a je presná.
5. **Co-registrácia** (`register_offset`) — deterministické hľadanie posunu masiek modelu
   voči mračnu s maximalizáciou IoU.
6. **Skrátenie hrany na reálny rozsah** — kontrola 9 vzoriek pozdĺž priesečnice, či obe
   roviny tam majú body (aby hrana „nevisela" v prázdne).

## Výsledok posledného behu (Trnava 7751/16A)

| Veličina | Hodnota |
|---|---|
| Roviny | 11 (sklony ~30°, 2 ploché) |
| Plocha strechy vs pôdorys | **351,4 m² vs 343,7 m² (1,02×)** |
| Presné hrany z priesečníc | **úžľabie 6, nárožie 7, hrebeň 4** |
| Odkvapové úseky | 15 |
| Podtypy rovín | plochá 2, vikier 3, sedlová/valbová 6 |
| Co-registrácia (IoU) | 0,756 (posun +0,25 / +0,25 m) |
| QA | WARN (0 chýb, 4 varovania o medzerách) |

**Porovnanie s tvojím popisom strechy:** 6 úžľabí ✓ (model 6), 4 hrebene ✓ (model 4),
3 trojuholníkové valby ✓ (model 3 vikiere/valby v podtypoch), 6 rovín od okapu —
model má 9 šikmých rovín, takže tu je ešte rozdiel (fragmentácia).

## Čo zostáva ako najväčší zdroj nepresnosti

1. **Presný obrys budovy** — OSM obrys je generalizovaný, ZBGIS vrstva neoverená;
   bez neho sa do modelu dostávajú časti susedných domov.
2. **Fragmentácia rovín** (9 šikmých plôch namiesto ~6). Riešenie: model trénovaný na
   **inštancie rovín** (datasety `data/roofs_mega*`), nie na typy striech.
3. **Časový nesúlad** — mračno 2018 vs ortofoto 2023–2025 (hlásené ako flag).
