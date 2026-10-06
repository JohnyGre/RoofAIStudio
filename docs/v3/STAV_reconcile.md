# Reconcile — finálna očista hrán (stav 2026-10-06)

`app/core/reconcile.py` nahrádza inline blok „4c" v `run_v3.py` (od Claude, s testami).

## Prečo vznikol

Inline blok 4c (môj) pri duplicite s **vlastnou** presnou hranou (X) hranu **zmazal celú**.
Na Beluji R1 to zmazalo polygónový hrebeň **14,5 m** a nechalo len 4,6 m X hranu —
pokrytie obvodu kleslo zo 100 % na 67 %.

## Oprava (reconcile_edges)

1. **Duplicita s vlastnou X hranou → OREZAŤ**, nie mazať: zostávajúce kusy ≥ 1 m si ponechajú
   pôvodný typ (a výšky). Kratšie kusy sa zahodia; ak X pokrýva celú hranu, hrana zmizne.
2. **`s` kus na presnej hrane INEJ roviny** → prevezme jej typ (spoločná hrana nie je štít).

Vracia štatistiku `{'trimmed','dropped','retyped'}`; je idempotentný a fail-soft.

## Overenie

- `tests/test_reconcile.py`: **14 OK, 0 FAIL** (14 m hrana + 4 m X → dva kusy 5 m + 5 m;
  úplné pokrytie → zmazanie; bez X hrán bez zmien; idempotencia; na reálnych fixture Triova/Beluj:
  pokrytie obvodu neklesne, QA 0/0, **Beluj R1 hrebeň > 13,5 m**).
- Reálne behy: Triova **PASS 0/0**, Beluj **PASS 0/0**, Átriová **PASS 0/0**;
  Beluj R1 hrebeň v kontrakte: `h4` 10,97 m + `Xh3` 4,59 m = **15,56 m**.
- `tools\ci.cmd` → **CI OK** (testy 13+16+24+31+14 = **98 kontrol**), regresia 3 adresy PASS.

## Zapojenie v run_v3.py

```python
from app.core import contract, eaves, engine, gis, ortho, preprocess, qa, reconcile, registration, vision
...
    # 4c) Finálna očista hrán nad kontraktom (po eaves): duplicita s vlastnou X hranou sa ORIEZNE
    #     (nie zmaže celá); 's' na presnej hrane inej roviny prevezme jej typ. Viď app/core/reconcile.py.
    try:
        _rc = reconcile.reconcile_edges(model)
        if any(_rc.values()):
            print(f"      očista hrán: {_rc}")
    except Exception as _ce:
        print("      očista hrán zlyhala:", _ce)
```

> Poznámka: lokálne commity `350b553` (eaves) a `b4abff8` (reconcile) čakajú na `git push`
> (credentials). Tento dokument + `reconcile.py` + `test_reconcile.py` sú nahraté cez konektor.
