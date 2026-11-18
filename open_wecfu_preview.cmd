@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\annotation\preview_wecfu.py"
if errorlevel 1 (
  echo Preview generation failed.
  pause
  exit /b 1
)
start "" "%~dp0outputs\wecfu_ultra_preview\index.html"
