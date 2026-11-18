@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\annotation\review_colonies.py"
if errorlevel 1 pause
