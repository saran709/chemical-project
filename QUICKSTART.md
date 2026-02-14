# Quick Start Guide - Automated Analysis

## Overview
This project now features a fully automated pipeline that processes new data and generates predictions with visualizations automatically.

## Method 1: Python Script (Recommended)

Simply run:
```bash
python src/autorun.py --input your_data.csv --temperature 25 --ph 7
```

**Example:**
```bash
python src/autorun.py --input data/my_experiment.csv --temperature 30 --ph 6
```

## Method 2: PowerShell Script (Windows)

```powershell
.\run_analysis.ps1 -InputFile data/your_data.csv -Temperature 25 -pH 7
```

**Example:**
```powershell
.\run_analysis.ps1 -InputFile data/my_experiment.csv -Temperature 30 -pH 6
```

## What Gets Generated

When you run the automated pipeline, it creates:

### 1. Predictions CSV
- **Location:** `reports/analysis/predictions.csv`
- Contains all input features plus predicted removal percentages

### 2. Main Visualizations (4 graphs)
- **Location:** `reports/analysis/visualizations/`
- `removal_vs_time_by_concentration.png` - How removal changes over time
- `removal_vs_concentration_by_time.png` - How removal changes with concentration
- `removal_heatmap.png` - Heatmap of time vs concentration
- `removal_3d_surface.png` - 3D surface plot

### 3. Comparison Analysis (7 graphs)
- **Location:** `reports/analysis/comparison/`
- `temperature_effect_heatmap.png` - Temperature effects across different conditions
- `ph_effect_heatmap.png` - pH effects across different conditions
- `temperature_effect_lines.png` - Temperature vs removal line plots
- `ph_effect_lines.png` - pH vs removal line plots
- `3d_comparison.png` - Dual 3D comparison plots
- `temperature_difference_heatmap.png` - Temperature difference from baseline
- `ph_difference_heatmap.png` - pH difference from baseline

### 4. Summary Report
- **Location:** `reports/analysis/ANALYSIS_REPORT.md`
- Complete analysis summary with statistics

## Input Data Format

Your input CSV/Excel should have these columns:
- `Time (min)`
- `Absorbance`
- `Concentration (mg/l)`
- `Amount adsorbed (mg/g)`

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | Required | Path to your CSV or Excel file |
| `--temperature` | 25 | Operating temperature in Celsius |
| `--ph` | 7 | Operating pH value |
| `--output` | reports/analysis | Output directory for all results |

## Examples

### Basic Usage (defaults: T=25°C, pH=7)
```bash
python src/autorun.py --input data/experiment_001.csv
```

### Custom Temperature and pH
```bash
python src/autorun.py --input data/experiment_002.csv --temperature 40 --ph 8
```

### Custom Output Directory
```bash
python src/autorun.py --input data/experiment_003.csv --output results/exp_003
```

## Processing Time

- Small datasets (< 2000 rows): ~30-60 seconds
- Medium datasets (2000-10000 rows): ~1-3 minutes
- Large datasets (> 10000 rows): ~3-10 minutes

## Troubleshooting

### "Input file not found"
- Check that the file path is correct
- Use forward slashes (/) or double backslashes (\\\\) in paths

### "Model not found"
- Ensure `models/removal_model.joblib` exists
- Run `python src/train.py --data model_data.xlsx` to train a model

### "Module not found" errors
- Activate the virtual environment: `.venv\Scripts\activate`
- OR install packages: `pip install -r requirements.txt`

## What's Next?

After the analysis completes:
1. Open the generated graphs to visualize results
2. Review the predictions CSV for detailed data
3. Check the ANALYSIS_REPORT.md for summary statistics
4. Compare temperature/pH effects using the difference heatmaps

## Tips

- **Multiple analyses:** Change the `--output` directory for each run to keep results separate
- **Batch processing:** Create a script to loop through multiple data files
- **Comparison:** Run with different temperature/pH values and compare outputs
- **Validation:** Check that prediction ranges make sense for your experimental domain

---

For advanced usage and manual control, see the main [README.md](README.md).
