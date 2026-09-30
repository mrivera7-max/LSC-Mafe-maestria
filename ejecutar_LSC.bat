@echo off
cd /d "%~dp0"
echo Ejecutando LSC-Mafe-maestria desde: %CD%
if not exist venv\Scripts\python.exe (
    echo No se encontro el entorno virtual "venv". Crealo con: python -m venv venv
    pause
    exit /b 1
)
venv\Scripts\python.exe app_unificada.py --comparar
pause
