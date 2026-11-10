@echo off
REM Opens GeoAI Detect in your web browser instead of its own window
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run install.bat first. & pause & exit /b 1)
.venv\Scripts\python.exe run_app.py --browser %*
