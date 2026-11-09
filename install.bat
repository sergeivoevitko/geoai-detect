@echo off
REM One-time setup of GeoAI Detect on Windows (needs Python 3.10-3.13 from python.org)
cd /d "%~dp0"
where py >/dev/null 2>/dev/null && (py -3 -m venv .venv) || (python -m venv .venv)
if not exist .venv\Scripts\python.exe (echo Python was not found. Install it from https://www.python.org/downloads/ and tick "Add to PATH". & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
REM For an NVIDIA GPU, first run:  .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Setup finished. Start the app with "GeoAI Detect.bat".
echo The AI models (about 6 GB) download automatically the first time they are used.
pause
