# Testovanie v3 — priebežný stav (regresné testy)

## Regresný test cez adresy (`tools/regression.py`)

Spúšťa celý v3 pipeline na **3 adresách** a overuje kritériá:
0 chýb kontraktu · QA bez chýb · ≥ 3 roviny · ≥ 3 klasifikované hrany · pomer plocha/pôdorys v [0,55; 1,8].

Posledný beh:

| Adresa | Čas | Rovín | Plocha m² | Pôdorys m² | hrany o/h/n/u/s | QA chyby | QA var | Pomer | PASS |
|---|---|---|---|---|---|---|---|---|---|
| Triova 7751/16A (Trnava) | 28 s | 11 | 286,9 | 343,7 | 10/10/21/15/15 | 0 | **0** | 0,83 | ✓ |
| Átriová 9309/16 (Trnava) | 73 s | 4 | 131,1 | — | 4/4/9/2/6 | 0 | **0** | — | ✓ |
| Beluj 50 (B. Štiavnica) | 74 s | 9 | 184,9 | — | 7/7/17/8/12 | 0 | 1 | — | ✓ |

**Celkový verdikt: PASS.**

## Čo testovanie odhalilo a čo som opravil

1. **Regresia sa zasekla na sieti** (Overpass timeouty pri sťahovaní obrysu) — oprava:
   cache sa kontroluje **pred** live sťahovaním (`gis.py::osm_footprint(cache_first=True)`)
   a timeouty sa skrátili na 15 s. Čas behu adresy klesol z „nikdy" na 28–74 s.
2. **Átriová mala medzeru 0,273 m** — bola tesne nad toleranciou snapu (0,25 m).
   Oprava: tolerancia 0,35 m a 5 iterácií → **varovanie zmizlo**.
3. **Beluj 50** má zvyšnú medzeru 0,263 m (1 varovanie) — plochy sú tam drobnejšie
   (malé vikierové plochy), takže pár vrcholov zostáva; QA chyby = 0.

## CI

`tools/ci.cmd` teraz beží v 5 krokoch: testy → selfcheck → QA gate → **regresia na 3 adresách** → export (DXF + HTML report).
Kroky vracajú nenulový kód pri zlyhaní, takže sa dá napojiť na akýkoľvek CI systém.

## Ďalšie kroky v testovaní

1. Pridať **meranie odchýlky hrán od ortofota** (máme skript na jednu adresu) ako metriku regresie.
2. Pridať adresy s inými typmi striech (plochá strecha, valbová) — teraz sú všetky tri sedlové/valbové.
3. Zladiť Belujovu zvyšnú medzeru (pravdepodobne iná logika snapu pre malé plochy).
