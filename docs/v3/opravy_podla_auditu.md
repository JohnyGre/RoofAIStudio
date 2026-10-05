# Opravy podľa auditu hrán (Átriová 9309/16, Triová, Beluj)

Audit z 2026-10-05 našiel veci, ktoré staré QA nezachytilo. Postupovalo sa presne
podľa navrhnutého poradia: **najprv QA kontroly, potom oprava zdrojov** — a nakoniec
overenie na všetkých troch adresách.

## 1. Nové QA kontroly (bez zmeny geometrie)

| Kontrola | Pravidlo | Nájde |
|---|---|---|
| `eave_horizontal` | odkvap musí byť takmer vodorovný (≤ 6°) | stúpajúca hrana omylom označená ako odkvap |
| `gable_shared` | štít (`s`) je voľná hrana — nesmie ležať na hrane inej roviny | nespárované spoločné hrany |
| `duplicate_edges` | tá istá fyzická hrana nesmie byť v kontrakte 2× | polygónová hrana + „X" z priesečnice |
| `edge_type_consistency` | typ hrany musí byť v oboch susedných rovinách rovnaký | protichodné typy |
| `plane_has_eave` | šikmá rovina musí mať odkvap | chýbajúca odkvapová hrana |
| `areas_true` | upozorni, že `area_m2` je pôdorys, nie plocha strechy | +11 % pri 26°, +16 % pri 30° |

Prvé spustenie na Átiovej 9309/16 **presne reprodukovalo nálezy auditu**: štít s2 na hrane
Xu2/n4, duplicity Xn1≈n4, Xh2≈h3, Xn3≈n4, a plochu 145,6 m² vs 131,1 m² (+11 %).

## 2. Opravy zdrojov

1. **`roof_outline_edges`** — dával obom koncom hrany **rovnakú výšku**, takže stúpajúca
   hrana vyzerala ako vodorovný odkvap. Teraz počíta výšku v oboch koncoch z vlastnej
   roviny a **odkvap (`o`) pridelí len vtedy, keď je hrana vodorovná (≤ 6°)**, inak je to štít.
2. **`classify_edges`** — pravidlo „najnižšia hrana = odkvap" dostalo navyše test vodorovnosti
   (audit: o4 kopíroval stúpajúcu hranu s2 z 158,15 na 161,33 m).
3. **Deduplikácia** — ak existuje presná hrana z priesečnice (X), polygónová hrana na tej istej
   čiare sa z kontraktu vyhodí (vrátane engine hrán s `exact=True`; kanonický je X záznam).
   Tým zmizol aj súčet 18 m hrebeňových záznamov pri reálnych ~8,2 m.
4. **Finálna konzistencia typov** (pred zostavením kontraktu):
   - `o` hrana, ktorá stúpa > 6° → pretypuje sa na `s`;
   - `s` hrana na presnej hrane inej roviny → **prevezme typ** tej presnej hrany (h/n/u);
   - dvojica `s` hrán na tej istej čiare → pretypuje sa podľa konvexnosti rovín (`_fold_convex`).
5. **Skutočná plocha** — kontrakt má nové pole `area_true_m2` (= pôdorys / cos(sklon));
   QA naň upozorňuje varovaním, pretože pre kalkuláciu materiálu je podstatný.

## 3. Výsledok (po opravách, deterministické behy)

| Adresa | Rovín | Hrany (o/h/n/u/s) | QA | Poznámka |
|---|---|---|---|---|
| Átriová 7751/16A (Triova) | 11 | podľa kontraktu | 0 chýb, 4 varovania | 2× `plane_has_eave` (chýbajúca odkvapová hrana), `areas_true` |
| Átriová 9309/16 | 4 | 15 | **0 chýb, 1 varovanie** | len `areas_true` (informatívne) |
| Beluj 50 | 9 | 30 | 0 chýb, 6 varovaní | `plane_has_eave` ×3, `areas_true` |

`tools\ci.cmd` → **CI OK** (testy + test mosta hrán + selfcheck + QA + regresia 3 adresy PASS + export).

## 4. Čo ešte zostáva (známe, hlásené ako varovania)

1. **Chýbajúce odkvapové hrany** (`plane_has_eave`) — po sprísnení pravidla sa ukázalo, že
   niektoré roviny majú vo svojom polygóne len „stúpajúce" hrany; ich skutočný odkvap bol
   pri orezávaní odstránený. Ďalší krok: doplniť odkvap z obrysu strechy pre tieto roviny.
2. **Odchýlka hrán voči ortofotu** 0,66–0,92 m — čaká na katastrálny obrys budovy (ZBGIS/MAPKA).
3. Inštančný segmentačný model (presné hranice plôch) — cesta bez externých dát cez
   vygenerované tréningové dáta z overených výstupov.
