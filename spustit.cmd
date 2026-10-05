@echo off
REM ============================================================
REM  RoofAIStudio v3 - spustac
REM  Pouzitie:  spustit.cmd [v3|testy|selfcheck|qa|foto|app|gui|vsetko]
REM ============================================================
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo CHYBA: nenasiel som virtualne prostredie "%PY%"
  echo Spusti najprv:  py -m venv .venv  a potom .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)

set CMD=%1
if "%CMD%"=="" set CMD=help

if /i "%CMD%"=="v3"        goto v3
if /i "%CMD%"=="v3bezfoto" goto v3bezfoto
if /i "%CMD%"=="testy"     goto testy
if /i "%CMD%"=="selfcheck" goto selfcheck
if /i "%CMD%"=="qa"        goto qa
if /i "%CMD%"=="foto"      goto foto
if /i "%CMD%"=="app"       goto app
if /i "%CMD%"=="gui"       goto gui
if /i "%CMD%"=="vsetko"    goto vsetko
goto help

:v3
echo === v3 pipeline: GPS -^> LiDAR -^> engine -^> kontrakt + QA + ORTOFOTO ===
"%PY%" tools\run_v3.py --ortho %2 %3 %4 %5 %6
exit /b %ERRORLEVEL%

:v3bezfoto
echo === v3 pipeline bez ortofota ===
"%PY%" tools\run_v3.py %2 %3 %4 %5 %6
exit /b %ERRORLEVEL%

:testy
echo === Testy jadra (kontrakt / QA / fuzia) ===
"%PY%" tests\test_core_contract.py
exit /b %ERRORLEVEL%

:selfcheck
echo === Overenie jadra na realnych datach ===
"%PY%" tools\core_selfcheck.py
exit /b %ERRORLEVEL%

:qa
set QA_FILE=%2
if "%QA_FILE%"=="" set QA_FILE=output\v3\Triova_7751_16A_Trnava_roofmodel_v3.json
echo === QA gate: %QA_FILE% ===
"%PY%" tools\qa_gate.py "%QA_FILE%"
exit /b %ERRORLEVEL%

:foto
set FOTO=%2
if "%FOTO%"=="" (
  echo Pouzitie: spustit.cmd foto "cesta\k\fotke.jpg" [conf]
  exit /b 2
)
echo === Fotka -^> detekcia -^> klasifikacia -^> GUI ===
"%PY%" tools\photo_pipeline.py "%FOTO%" %3
exit /b %ERRORLEVEL%

:app
echo === Desktop aplikacia (PySide6) ===
"%PY%" roofai_desktop_v2.py
exit /b %ERRORLEVEL%

:gui
echo Otvaram posledne vygenerovane GUI...
for %%F in (output\photo_pipeline\*_pipeline_gui.html) do set LASTGUI=%%F
if defined LASTGUI ( start "" "%LASTGUI%" ) else ( echo Ziadne GUI zatial nie je - spusti najprv: spustit.cmd foto "foto.jpg" )
exit /b 0

:vsetko
echo === 1/3 testy ===
"%PY%" tests\test_core_contract.py || exit /b 1
echo === 2/3 selfcheck ===
"%PY%" tools\core_selfcheck.py
echo === 3/3 QA gate ===
"%PY%" tools\qa_gate.py "output\v3\Triova_7751_16A_Trnava_roofmodel_v3.json"
exit /b %ERRORLEVEL%

:help
echo.
echo   spustit.cmd v3               - v3 pipeline + ortofoto v max. rozliseni
echo   spustit.cmd v3bezfoto        - v3 pipeline bez ortofota (rychlejsie)
echo   spustit.cmd v3 --address "Atriova 9309/16, Trnava"
echo   spustit.cmd testy            - testy jadra (13 kontrol)
echo   spustit.cmd selfcheck        - overenie jadra na realnych datach
echo   spustit.cmd qa [subor.json]  - QA gate nad vystupom pipeline
echo   spustit.cmd foto "f.jpg" 0.15 - fotka -^> detekcia -^> klasifikacia -^> GUI
echo   spustit.cmd app              - desktop aplikacia RoofAIStudio
echo   spustit.cmd gui              - otvor posledne vygenerovane HTML GUI
echo   spustit.cmd vsetko           - testy + selfcheck + QA gate
echo.
exit /b 0
