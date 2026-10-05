@echo off
REM ============================================================
REM  RoofAIStudio v3 - CI: testy + selfcheck + QA + regresia + export
REM  Pouzitie:  ci.cmd
REM ============================================================
setlocal
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (echo CHYBA: chyba venv & exit /b 1)

echo === 1/5 testy jadra ===
"%PY%" tests\test_core_contract.py || exit /b 1
echo   (test mosta hran desktop-^>kontrakt)
"%PY%" tests\test_edges_bridge.py || exit /b 1
echo   (test QA kontrol hrán)
"%PY%" tests\test_edge_qa.py || exit /b 1

echo === 2/5 selfcheck (kontrakt/fuzia/registracia/QA) ===
"%PY%" tools\core_selfcheck.py || exit /b 1

echo === 3/5 QA gate (hlavna adresa) ===
"%PY%" tools\qa_gate.py "output\v3\Triova_7751_16A_Trnava_roofmodel_v3.json" || exit /b 1

echo === 4/5 regresia na 3 adresach ===
"%PY%" tools\regression.py || exit /b 1

echo === 5/5 export DXF + HTML report ===
"%PY%" tools\v3_export.py "output\v3\Triova_7751_16A_Trnava_roofmodel_v3.json" || exit /b 1

echo.
echo CI OK
exit /b 0
