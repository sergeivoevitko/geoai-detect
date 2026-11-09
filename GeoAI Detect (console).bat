@echo off
REM Same as "GeoAI Detect.bat" but keeps a console window open with log messages (for troubleshooting)
cd /d "%~dp0"
.venv\Scripts\python.exe run_app.py %*
pause
