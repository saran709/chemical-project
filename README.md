# Machine Learning Based Prediction and Analysis of Adsorption Removal Efficiency

## Project overview
This project builds a supervised regression model to predict adsorption removal efficiency from experimental process parameters. The system is designed to help analyze adsorption performance, reduce experimental cost, and support data-driven optimization of operating conditions.

## Dataset schema
The preferred dataset schema is:

- time
- dosage of adsorbent
- temperature
- PH
- Adsorbent Concentration
- removal

The current training pipeline also supports the columns present in model_data.xlsx:

- Time (min)
- Absorbance
- Concentration (mg/l)
- Amount adsorbed (mg/g)
- % removal

## Pipeline summary
- Data loading from CSV or XLSX
- Data cleaning with median imputation
- Optional normalization for linear and neural models
- Multiple regression models evaluated:
	- Linear Regression
	- Random Forest Regression
	- Gradient Boosting Regression
	- Artificial Neural Network (MLP)
- Evaluation with MAE, RMSE, and R2
- Feature importance analysis
- Model persistence with feature metadata
- Reports written to the reports folder

## Quick Start - Automated Pipeline

The fastest way to get predictions and visualizations is using the automated pipeline:

### Windows (Batch File)
```cmd
run_analysis.bat data/your_data.csv [temperature] [ph]
```

Examples:
```cmd
# Default: T=25°C, pH=7
run_analysis.bat data/my_data.csv

# Custom: T=30°C, pH=6
run_analysis.bat data/my_data.csv 30 6
```

### Windows (PowerShell)
```powershell
.\run_analysis.ps1 -InputFile data/your_data.csv [-Temperature 25] [-pH 7]
```

Examples:
```powershell
# Default settings
.\run_analysis.ps1 -InputFile data/my_data.csv

# Custom settings
.\run_analysis.ps1 -InputFile data/my_data.csv -Temperature 30 -pH 6
```

### Manual Python Command
```bash
python src/autorun.py --input data/your_data.csv --temperature 25 --ph 7
```

### Output Generated
The automated pipeline creates:
- **predictions/predictions.csv** - All predictions with features
- **reports/visualizations/** - Main analysis graphs:
  - `removal_vs_time_by_concentration.png`
  - `removal_vs_concentration_by_time.png`
  - `removal_heatmap.png`
  - `removal_3d_surface.png`
- **reports/comparison/** - Comparison analysis across all conditions:
  - `temperature_effect_heatmap.png`
  - `ph_effect_heatmap.png`
  - `temperature_effect_lines.png`
  - `ph_effect_lines.png`
  - `3d_comparison.png`
  - `temperature_difference_heatmap.png`
  - `ph_difference_heatmap.png`
- **ANALYSIS_REPORT.md** - Summary report

## Advanced Usage - Manual Steps

For more control over the pipeline:

1) Place your dataset in the project folder.
2) Install dependencies from requirements.txt.
3) Train the model:
	- `python src/train.py --data path_to_dataset --output models/removal_model.joblib --report-dir reports`
4) Run predictions and visualization:
	- `python src/predict_and_visualize.py --temperature 25 --ph 7`
5) Run comparison analysis:
	- `python src/compare_conditions.py`

The prediction output includes all feature columns plus a `predicted_removal` column.

## Reports
After training, the following files are written to the reports folder:

- model_metrics.csv (MAE, RMSE, R2 for each model)
- feature_importance.csv (best model feature influence)
- correlation_matrix.csv (EDA correlations)

## Notes
- If your dataset uses the preferred schema, the model will train directly.
- If your dataset uses the model_data.xlsx schema, the model will train using that column set.
- Extend src/train.py to evaluate additional models if needed.
