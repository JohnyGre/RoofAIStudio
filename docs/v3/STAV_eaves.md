# Stav zapojenia odkvapov (eaves) — 2026-10-06 po integrácii files4

Lokálny stav: commit **350b553** (obsahuje všetko nižšie; push na GitHub čaká na credentials).
Tento dokument je v repozitári pre prípad, že by kód ešte nebol pushnutý.

## Výsledok po integrácii Claudeho verzie (files4.zip)

| Adresa | QA | Odkvapy |
|---|---|---|
| Átriová 9309/16 | **PASS 0/0** | — |
| Átriová 7751/16A | **PASS 0/0** | R4.o1 promoted (11,19 m), R5.o1 promoted (5,23 m) |
| Beluj 50 | **PASS 0/0** | R4.o1 added (11,93 m), R9.o1 promoted (6,15 m) |

Testy: `test_core_contract` 13 OK · `test_edges_bridge` 16 OK · `test_edge_qa` **24 OK** ·
`test_eaves` **31 OK** · CI OK (vrátane regresie 3 adresy).

## Čo obsahuje commit 350b553 (ak treba dorovnať ručne)

- `app/core/eaves.py` — nový (quarantine_degenerate_planes + add_missing_eaves s povýšením s→o;
  fallback „úplne stratený odkvap“ je popísaný v hlavičke modulu ako ďalší krok)
- `app/core/qa.py` — 24 kontrol; severity: štít na cudzej hrane / stúpajúci odkvap / typová
  nekonzistencia = **chyby**, malé strmé roviny bez odkvapu = **info**, plocha = **info**
- `tools/run_v3.py` — `eaves.quarantine_degenerate_planes` + `eaves.add_missing_eaves` pred QA;
  navyše **finálna očista** po eaves (s na presnej hrane → prevezme typ; duplicita k vlastnej X → preč)
- `tests/test_eaves.py`, `tests/test_edge_qa.py` — rozšírené (31 a 24 kontrol)
- `tests/fixtures/roof_11planes_before_eaves.json`, `roof_9planes_steep_before_eaves.json`,
  `roof_hip_valley_before_fix.json`
- `tools/ci.cmd` — spúšťa všetky štyri testovacie sady
- `docs/v3/priklady/` — obnovené kontrakty a QA reporty (všetky PASS 0/0)

## Jedna vec na dorovnanie

`git push origin master` v `C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio`
(Git Credential Manager vypýta prihlásenie v interaktívnom okne).
