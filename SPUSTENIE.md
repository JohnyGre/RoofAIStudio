# Ako to spustiť (RoofAIStudio v3)

Všetko sa spúšťa z koreňa projektu `C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio`
a **vždy cez venv** (`.venv\Scripts\python.exe`) — systémový Python nemá torch ani ultralytics.

## Najjednoduchšie: spúšťač

```cmd
cd C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio
spustit.cmd                :: zobrazí ponuku
spustit.cmd testy          :: testy jadra (13 kontrol)
spustit.cmd selfcheck      :: overenie jadra na reálnych dátach
spustit.cmd qa             :: QA gate nad posledným v3 výstupom
spustit.cmd foto "C:\cesta\k\fotke.jpg" 0.15
spustit.cmd app            :: desktop aplikácia (PySide6)
spustit.cmd gui            :: otvorí posledné vygenerované HTML GUI
spustit.cmd vsetko         :: testy + selfcheck + QA gate
```

## Ručne (rovnaké príkazy, bez spúšťača)

```powershell
cd C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio

# 1) testy nového jadra
.venv\Scripts\python.exe tests\test_core_contract.py

# 2) overenie jadra na reálnych dátach (výstup: output\core_checks\report.json)
.venv\Scripts\python.exe tools\core_selfcheck.py

# 3) QA gate nad v3 kontraktom (exit kód 1 = neprešlo)
.venv\Scripts\python.exe tools\qa_gate.py output\_triov__7751_16A__917_01_Trnava_roofmodel_v3.json

# 4) fotka → 640×640 → detekcia → klasifikácia → HTML GUI
.venv\Scripts\python.exe tools\photo_pipeline.py "data\ortho\manual_ortho.jpg" 0.15

# 5) desktop aplikácia (adresa → LiDAR → roviny → 3D viewer)
.venv\Scripts\python.exe roofai_desktop_v2.py
```

## Kde sú výsledky

| Čo | Kde |
|---|---|
| HTML GUI (vrstvy + tabuľka detekcií) | `output\photo_pipeline\<nazov>_pipeline_gui.html` |
| Vrstvy ako PNG | `output\photo_pipeline\<nazov>_overlay.png`, `_processed_640.png` |
| v3 kontrakt + QA report | `output\<nazov>_roofmodel_v3.json`, `..._qa.json` |
| Protokol overenia jadra | `output\core_checks\report.json` |
| SQLite log behov | `output\photo_pipeline\pipeline_results.db` |

## Ak niečo nefunguje

| Príznak | Príčina / riešenie |
|---|---|
| `No module named 'torch'` | spúšťaš systémovým Pythonom → použi `.venv\Scripts\python.exe` |
| `No space left on device` | diskový priestor (C: bol plný) |
| Prázdne detekcie | zníž prah: `spustit.cmd foto "f.jpg" 0.10` |
| QA gate vracia 1 | otvor `..._qa.json` — sú tam konkrétne chyby |
| GUI sa otvorí bez obrázkov | otvor HTML dvojklikom (obrázky sú vložené priamo v ňom) |
