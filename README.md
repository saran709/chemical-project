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

## Usage
1) Place your dataset in the project folder.
2) Install dependencies from requirements.txt.
3) Train the model:
	- `python src/train.py --data path_to_dataset --output models/removal_model.joblib --report-dir reports`
4) Run predictions:
	- `python src/predict.py --model models/removal_model.joblib --input path_to_input --output predictions/predictions.csv`

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
