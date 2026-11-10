@echo off
REM One-time setup of GeoAI Detect on Windows. Needs Python 3.12 from python.org
cd /d "%~dp0"
set PY=
py -3.12 -c "import sys" >nul 2>&1 && set PY=py -3.12
if not defined PY (py -3.11 -c "import sys" >nul 2>&1 && set PY=py -3.11)
if not defined PY (python -c "import sys; assert (3,10)<=sys.version_info[:2]<=(3,13)" >nul 2>&1 && set PY=python)
if not defined PY (echo Python 3.12 was not found. Install it from https://www.python.org/downloads/release/python-3128/ and tick "Add python.exe to PATH". & pause & exit /b 1)
echo Using %PY%
%PY% -m venv .venv
if not exist .venv\Scripts\python.exe (echo Could not create the virtual environment. & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
REM NVIDIA GPU only: uncomment the next line to install the GPU version of PyTorch first
REM .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (echo. & echo Installation failed. Send the messages above. & pause & exit /b 1)
echo.
echo Setup finished. Start the app with "GeoAI Detect.bat".
echo The AI models (about 6 GB) download automatically the first time a detection runs.
pause
