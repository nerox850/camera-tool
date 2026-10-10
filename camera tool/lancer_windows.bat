@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo Python 3.9 a 3.12 est requis. Installez Python depuis https://www.python.org/downloads/
    echo puis relancez ce fichier.
    pause
    exit /b 1
)

set "PYTHON="
py -3.12 --version >nul 2>&1
if not errorlevel 1 set "PYTHON=py -3.12"
if not defined PYTHON (
    py -3.11 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3.11"
)
if not defined PYTHON (
    py -3.10 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3.10"
)
if not defined PYTHON (
    py -3.9 --version >nul 2>&1
    if not errorlevel 1 set "PYTHON=py -3.9"
)
if not defined PYTHON (
    echo Python 3.9 a 3.12 est requis. Installez une version compatible.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
    goto :install
)

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>&1
if errorlevel 1 (
    echo L'environnement .venv utilise une version incompatible de Python.
    echo Supprimez uniquement le dossier .venv puis relancez ce fichier.
    pause
    exit /b 1
)

:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

".venv\Scripts\python.exe" dessin_camera.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Le lancement a echoue. Verifiez votre connexion Internet, Python 3 et l'acces a la webcam.
pause
exit /b 1
