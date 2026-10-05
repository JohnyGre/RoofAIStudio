# Finálny stav: hrany strechy z v3 pipeline (Trnava 7751/16A)

## Čo je hotové (podľa tvojho návrhu)

Tvoj postreh z farebnej mapy plôch („stačí tie body ohraničiť pravidelnými tvarmi a máme roviny")
je v kóde zavedený: `regularize_polygon()` ohraničí každý zhluk **minimálnym rotovaným
obdĺžnikom** alebo **minimálnym trojuholníkom** (`cv2.minAreaRect` / `cv2.minEnclosingTriangle`),
a to len vtedy, keď tvar sedí (≤ 1,10× obal). Plochy tak majú 3–4 rohy namiesto zubatých obrysov.
Hrany potom vznikajú z ich spoločných línií.

## Úplný reťaz, ktorý sa na výpočet používa

| Krok | Zdroj / metóda | Výsledok |
|---|---|---|
| Oblasť | LAZ, trieda 6 + výškový pás ±7 m + voxel 0,15 m | 12 266 → 10 205 bodov |
| Pôdorys | GIS obrys (OSM, s geometriou) + obrys z vycvičeného modelu | orezanie na jednu budovu |
| Registrácia | deterministická (max IoU) medzi modelom a mračnom | posun (+0,25, +0,25) m, **IoU 0,756** |
| Roviny | RANSAC → CC split → zlúčenie kolineárnych → orezanie pravidelným tvarom | **11 rovín** |
| Hrany hrebeň/nárožie/úžľabie | presné **priesečnice** + konvexnosť z **reálnych bodov** mračna | **6 úžľabí, 4 hrebene, 7 nárožných hrán** |
| Odkvap | hrana na vonkajšom obvode (za ňou už nie sú body) | **15 úsekov** |
| QA | kontrakt + geometrické kontroly | **0 chýb** (4 varovania o medzerách < 0,25 m) |

Podtypy rovín: plochá 2, vikier 3, sedlová/valbová 5, trojuholníková valba 1.
Plocha strechy 286,6 m², pôdorys 343,7 m².

## Objektívne meranie presnosti (bez vizuálneho modelu)

Pre každú nakreslenú hranu sa v 15 bodoch hľadá najtmavšia línia v ortofote (skutočná hrana)
kolmo na hranu a meria sa odchýlka v metroch:

| Typ | Hrán | Priemerná odchýlka | Medián | Maximum |
|---|---|---|---|---|
| nárožie | 19 | 0,92 m | 0,90 m | 1,16 m |
| úžľabie | 6 | 0,66 m | 0,53 m | 1,19 m |
| odkvap | — | nemerané (málo výrazných línií) | — | — |
| hrebeň | — | nemerané | — | — |

## Vizuálna kontrola (niekoľko kôl, externý vision model)

Naposledy: odkvapy „prevažne na okraji", hrebeň „čiastočne sedí", nárožia a úžľabia
„posunuté"; celkovo **použiteľné s čiastočnými chybami**. Hodnotenie sa medzi kolami líši
(ten istý typ obrázka raz „dobrý s malými chybami", raz „málo presný"), preto je vizuálny
model len orientačný — rozhodujúce je číselné meranie vyššie.

## Čo skúšané bolo a nefungovalo (aby sa to neopakovalo)

- segmentácia riadená maskami modelu → model delí **typy** striech, nie roviny (sklony 25–65°),
- konvexnosť z extrapolácie rovín → systematicky nesprávna (raz všetko hrebeň, raz všetko úžľabie),
- globálny posun modelu na línie ortofota → odchýlka sa nezmenšila (0,52 → 0,75 m),
- iterovaný refit v rámci vlastného obrysu → plochy sa zúžili až na 0 m² a hrany zmizli.

## Čo by dostalo presnosť na centimetre

1. **Presný obrys budovy** (ZBGIS katastrálna vrstva) — bez neho sa do modelu miešajú časti susedov.
2. **Model trénovaný na inštancie rovín** (nie na typy) — potom by masky definovali plochy
   a mračno len výšky; datasety `data/roofs_mega*` už v projekte sú.
3. Prípadne umelecká rovina: prispôsobiť hrany líniám ortofota optimalizáciou (nie globálnym posunom).
