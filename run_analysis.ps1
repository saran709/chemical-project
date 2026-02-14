# Automated Adsorption Removal Analysis Pipeline
# Usage: ./run_analysis.ps1 -InputFile <path> [-Temperature <value>] [-pH <value>]

param(
    [Parameter(Mandatory=$true, HelpMessage="Path to input CSV or Excel file")]
    [string]$InputFile,
    
    [Parameter(HelpMessage="Temperature in Celsius (default: 25)")]
    [float]$Temperature = 25,
    
    [Parameter(HelpMessage="pH value (default: 7)")]
    [float]$pH = 7
)

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

# Verify input file exists
if (-not (Test-Path $InputFile)) {
    Write-Host "[ERROR] Input file not found: $InputFile" -ForegroundColor Red
    exit 1
}

Write-Header "AUTOMATED ADSORPTION REMOVAL ANALYSIS PIPELINE"

Write-Host "Input File: $InputFile" -ForegroundColor Green
Write-Host "Temperature: $($Temperature)C" -ForegroundColor Green
Write-Host "pH: $pH" -ForegroundColor Green
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $PSScriptRoot
if (-not $scriptDir) { $scriptDir = $PSScriptRoot }
Set-Location $scriptDir

# Run the automated pipeline
Write-Host "Starting analysis..." -ForegroundColor Yellow
& ".\.venv\Scripts\python.exe" src\autorun.py --input "$InputFile" --temperature $Temperature --ph $pH

if ($LASTEXITCODE -eq 0) {
    Write-Header "[SUCCESS] ANALYSIS SUCCESSFUL!"
    Write-Host "All results have been generated. Check the output directory for details." -ForegroundColor Green
} else {
    Write-Header "[ERROR] ANALYSIS FAILED!"
    Write-Host "An error occurred during analysis. Please check the output above." -ForegroundColor Red
    exit 1
}
