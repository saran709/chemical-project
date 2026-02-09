import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STANDARD_FEATURES = [
    "time",
    "dosage of adsorbent",
    "temperature",
    "PH",
    "Adsorbent Concentration",
]
STANDARD_TARGET = "removal"

ALT_FEATURES = [
    "Time (min)",
    "Absorbance",
    "Concentration (mg/l)",
    "Amount adsorbed (mg/g)",
]
ALT_TARGET = "% removal"


def _lower_map(columns):
    return {str(col).strip().lower(): col for col in columns}


def _resolve_columns(df: pd.DataFrame):
    lower_map = _lower_map(df.columns)

    def resolve(cols):
        resolved = []
        for col in cols:
            key = str(col).strip().lower()
            if key not in lower_map:
                return None
            resolved.append(lower_map[key])
        return resolved

    standard_features = resolve(STANDARD_FEATURES)
    standard_target = resolve([STANDARD_TARGET])
    if standard_features and standard_target:
        return standard_features, standard_target[0], "standard"

    alt_features = resolve(ALT_FEATURES)
    alt_target = resolve([ALT_TARGET])
    if alt_features and alt_target:
        return alt_features, alt_target[0], "alternate"

    raise ValueError(
        "Dataset columns do not match expected schema. "
        "Provide columns: "
        + ", ".join(STANDARD_FEATURES + [STANDARD_TARGET])
        + " or: "
        + ", ".join(ALT_FEATURES + [ALT_TARGET])
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


def build_preprocessor(feature_cols, scale):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=steps)

    return ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)]
    )


def build_models(feature_cols):
    return {
        "LinearRegression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_cols, scale=True)),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForestRegressor": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_cols, scale=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
        "GradientBoostingRegressor": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_cols, scale=False)),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
        "MLPRegressor": Pipeline(
            steps=[
                ("preprocess", build_preprocessor(feature_cols, scale=True)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return mae, rmse, r2


def compute_feature_importance(model, feature_cols, X_test, y_test):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importances = np.abs(coef)
    else:
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="r2",
        )
        importances = result.importances_mean

    return pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)


def train(args):
    data_path = Path(args.data)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)

    df = load_dataset(data_path)
    feature_cols, target_col, schema = _resolve_columns(df)
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = build_models(feature_cols)
    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        mae, rmse, r2 = evaluate_model(model, X_test, y_test)
        results.append(
            {
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )
        trained[name] = model

    results_df = pd.DataFrame(results).sort_values(by="rmse")
    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": best_model,
        "features": feature_cols,
        "target": target_col,
        "model_name": best_name,
        "metrics": results_df.to_dict(orient="records"),
    }
    joblib.dump(payload, output_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_dir / "model_metrics.csv", index=False)
    feature_importance = compute_feature_importance(
        best_model, feature_cols, X_test, y_test
    )
    feature_importance.to_csv(
        report_dir / "feature_importance.csv", index=False
    )
    corr = df[feature_cols + [target_col]].corr(numeric_only=True)
    corr.to_csv(report_dir / "correlation_matrix.csv")

    print(f"Model saved to: {output_path}")
    print(f"Schema used: {schema}")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Best model: {best_name}")
    print("Metrics:")
    print(results_df.to_string(index=False))
    print(f"Reports saved to: {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train removal prediction model")
    parser.add_argument(
        "--data",
        required=True,
        help=(
            "Path to CSV/XLSX dataset with columns: time, dosage of adsorbent, "
            "temperature, PH, Adsorbent Concentration, removal"
        ),
    )
    parser.add_argument(
        "--output",
        default="models/removal_model.joblib",
        help="Path to save trained model",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to write metrics and feature importance",
    )
    train(parser.parse_args())
