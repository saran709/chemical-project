# Automated Adsorption Removal Analysis System

## Project Status: FULLY AUTOMATED ✓

This project now features complete automation for processing new data and generating predictions with comprehensive visualizations.

## Quick Commands

### Fastest Way to Run
```bash
# Windows PowerShell
python src/autorun.py --input your_data.csv

# With custom parameters
python src/autorun.py --input your_data.csv --temperature 30 --ph 6
```

## Complete File Structure

```
chemical-project/
├── run_analysis.bat          # Windows batch automation script
├── run_analysis.ps1          # Windows PowerShell automation script  
├── QUICKSTART.md            # Quick start guide
├── README.md                # Full documentation
│
├── src/
│   ├── autorun.py           # Main automation pipeline ⭐
│   ├── predict_and_visualize.py  # Predictions + visualizations
│   ├── compare_conditions.py     # Comparison analysis
│   ├── train.py             # Model training
│   └── predict.py           # Simple predictions
│
├── models/
│   └── removal_model.joblib # Trained ML model
│
├── data/
│   └── your_data_files.csv  # Input data goes here
│
└── reports/
    └── analysis/            # Output directory (auto-created)
        ├── predictions.csv  # Prediction results
        ├── visualizations/  # Main graphs (4 files)
        ├── comparison/      # Comparison graphs (7 files)
        └── ANALYSIS_REPORT.md  # Summary report
```

## Automated Outputs (11 Total Graphs)

### Main Analysis (4 graphs)
1. removal_vs_time_by_concentration.png
2. removal_vs_concentration_by_time.png
3. removal_heatmap.png
4. removal_3d_surface.png

### Comparison Analysis (7 graphs)
5. temperature_effect_heatmap.png
6. ph_effect_heatmap.png
7. temperature_effect_lines.png
8. ph_effect_lines.png
9. 3d_comparison.png
10. temperature_difference_heatmap.png
11. ph_difference_heatmap.png

## Key Features

✓ **One-command execution** - Single command generates everything
✓ **Automatic data validation** - Checks input file format
✓ **Smart defaults** - Uses T=25°C, pH=7 if not specified
✓ **Multiple visualizations** - 11 different analysis graphs
✓ **Comparison analysis** - Shows how conditions affect removal
✓ **Summary reports** - Auto-generates markdown reports
✓ **Error handling** - Clear error messages if something fails
✓ **Flexible I/O** - Specify custom input/output directories

## Usage Examples

### 1. Quick Analysis (Default Settings)
```bash
python src/autorun.py --input data/experiment.csv
```
→ Uses T=25°C, pH=7, outputs to `reports/analysis/`

### 2. Custom Conditions
```bash
python src/autorun.py --input data/experiment.csv --temperature 35 --ph 8
```
→ Uses T=35°C, pH=8

### 3. Custom Output Directory
```bash
python src/autorun.py --input data/exp_001.csv --output results/exp_001
```
→ Saves all results to `results/exp_001/`

### 4. Full Custom
```bash
python src/autorun.py \
    --input data/my_data.csv \
    --temperature 40 \
    --ph 6.5 \
    --output results/analysis_2026_02_10
```

## Workflow

```
[New Data File] 
    ↓
[autorun.py]
    ↓
├─→ [predict_and_visualize.py] → Main predictions & 4 graphs
    ↓
├─→ [compare_conditions.py] → Comparison analysis & 7 graphs
    ↓
└─→ [create_summary_report()] → ANALYSIS_REPORT.md
    ↓
[ALL RESULTS READY]
```

## Processing Time

| Dataset Size | Time Required |
|--------------|---------------|
| < 2,000 rows | 30-60 seconds |
| 2,000-10,000 rows | 1-3 minutes |
| > 10,000 rows | 3-10 minutes |

## Dependencies

All automatically handled by the virtual environment:
- pandas - Data manipulation
- scikit-learn - ML model
- matplotlib - Visualization
- seaborn - Statistical plots
- joblib - Model persistence
- numpy - Numerical computing

## Input Format

Expected CSV/Excel columns:
- Time (min)
- Absorbance
- Concentration (mg/l)
- Amount adsorbed (mg/g)

## Output Format

### Predictions CSV
| Time | Absorbance | Concentration | Amount adsorbed | predicted_removal | temperature | PH |
|------|------------|---------------|-----------------|-------------------|-------------|----|
| 10   | 0.01       | 50.0          | 7.0             | 86.65             | 25.0        | 7.0|

### Statistics
- Mean removal percentage
- Min/max removal values
- Standard deviation
- Predictions grouped by conditions

## Advanced Usage

### Batch Processing Multiple Files
```bash
# Process all CSV files in a directory
for file in data/*.csv; do
    python src/autorun.py --input "$file" --output "results/$(basename $file .csv)"
done
```

### Compare Different Conditions
```bash
# Run same data at different temperatures
python src/autorun.py --input data/exp.csv --temperature 20 --output results/T20
python src/autorun.py --input data/exp.csv --temperature 30 --output results/T30
python src/autorun.py --input data/exp.csv --temperature 40 --output results/T40
```

## Benefits

1. **Time Saving** - Generate all analysis in one command
2. **Consistency** - Same analysis process every time
3. **Reproducibility** - All parameters logged in report
4. **Comprehensive** - 11 different visualization perspectives
5. **User-Friendly** - No coding knowledge required
6. **Automated QA** - Built-in validation and error checking

## Next Steps

1. Place your data file in the `data/` directory
2. Run `python src/autorun.py --input data/your_file.csv`
3. Wait 1-3 minutes for processing
4. Check `reports/analysis/` for all results
5. Review graphs and summary report
6. Analyze predictions CSV for detailed data

## Support

For issues or questions:
1. Check QUICKSTART.md for common solutions
2. Review README.md for detailed documentation
3. Inspect terminal output for error messages
4. Verify input data format matches expected schema

---

**System Ready:** Just provide your data file and run! 🚀
