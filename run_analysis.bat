@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo ============================================================
    echo Automated Adsorption Removal Analysis Pipeline
    echo ============================================================
    echo.
    echo Usage: run_analysis.bat input_file [temperature] [ph]
    echo.
    echo Arguments:
    echo   input_file      - Path to CSV or Excel file with data (required)
    echo   temperature     - Temperature in Celsius (default: 25)
    echo   ph              - pH value (default: 7)
    echo.
    echo Examples:
    echo   run_analysis.bat data\my_data.csv
    echo   run_analysis.bat data\my_data.csv 30 6
    echo   run_analysis.bat data\my_data.csv 40 8
    echo.
    exit /b 1
)

set INPUT_FILE=%~1
set TEMPERATURE=%2
set PH=%3

if "!TEMPERATURE!"=="" set TEMPERATURE=25
if "!PH!"=="" set PH=7

cd /d "%~dp0"

.venv\Scripts\python.exe src\autorun.py --input "!INPUT_FILE!" --temperature !TEMPERATURE! --ph !PH!
