# Pre Claude — výsledok integrácie tvojej verzie (files4) — 2026-10-06

## Integrované bez zmien
`eaves.py` (quarantine + add_missing_eaves s povýšením s→o), `qa.py` (24 kontrol),
`test_eaves.py` (31 kontrol), `test_edge_qa.py` (24 kontrol), fixtures
(`roof_11planes_before_eaves.json`, `roof_9planes_steep_before_eaves.json`, `roof_hip_valley_before_fix.json`)
a tvoj `run_v3.py` (volanie `quarantine_degenerate_planes` + `add_missing_eaves` pred QA).

## Výsledok na reálnych adresách — **všetky tri PASS 0/0**

| Adresa | QA | Čo tvoj kód urobil |
|---|---|---|
| Átriová 9309/16 | **PASS (0/0)** | — (bolo čisté) |
| Átriová 7751/16A | **PASS (0/0)** | R4.o1 promoted (11,19 m), R5.o1 promoted (5,23 m); R1 degenerovaná → quarantine |
| Beluj 50 | **PASS (0/0)** | R4.o1 added (11,93 m), R9.o1 promoted (6,15 m); R5 degenerovaná → quarantine |

Testy: 13 + 16 + **24** + **31** = 84 OK, 0 FAIL. `tools\ci.cmd` → **CI OK**.
Regresný report: 3 adresy PASS, všade `errors=0, warnings=0` (info: len plocha / malé strmé roviny).

## Jedna vec, ktorú som pridal za tvoj kód (a prečo)

Po `eaves` zostávali na Triovej dva nálezy (ktoré tvoj modul nemal ako odstrániť):
`gable_shared: štít R9.s1 leží na hrane R3.Xn4` a `duplicate_edges: R9 s1 ≈ Xn2`.
Je to hrana, ktorá leží **na presnej hrane inej roviny** (a aj na vlastnej X hrane), ale bola
typu `s` — teda spoločná hrana vydávaná za štít.

Doplnil som preto do `run_v3.py` malý blok **„4c) Finálna očista hrán nad kontraktom"**
(po tvojom `add_missing_eaves`, pred QA, fail-soft):

1. `s` hrana, ktorá koinciduje (`qa._coincide`) s presnou hranou **inej** roviny → prevezme jej typ;
2. hrana (nie X), ktorá koinciduje s **vlastnou** X hranou roviny → odstráni sa (duplicita).

Po tomto bode: Triova aj Beluj **0 chýb / 0 varovaní**.

## Tvoje „obmedzenie" už nie je potrebné

V hlavičke `eaves.py` máš: „odkvap, ktorý polygón stratil úplne, sa tu nenájde — treba
`engine.roof_outline_ring`". Na našich troch adresách to **netreba**: po tvojom quarantine
degenerovaných rovín (ktoré vyrábali vodorovné „úžľabie" na čiare odkvapu) sa ukázalo,
že skutočný odkvap **bol medzi stranami polygónu** vo všetkých prípadoch — len bol vyhodený
deduplikáciou proti tej degenerovanej priesečnici. Fallback cez `roof_outline_ring` teda
nechaj v zálohe (dobré mať), ale na overených dátach nie je potrebný.

## Poznámka k repozitáru

Všetko je v lokálnom commite `350b553`. Push na GitHub zlyháva na credentials
(Git Credential Manager v tejto relácii nemá terminál) — `eaves.py` + stavový dokument
sú nahraté cez konektor (commit `13c8949`), zvyšok dorovná používateľ jedným `git push`.
