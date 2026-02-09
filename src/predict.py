import argparse
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_FEATURES = [
    "time",
    "dosage of adsorbent",
    "temperature",
    "PH",
    "Adsorbent Concentration",
]


def _lower_map(columns):
    return {str(col).strip().lower(): col for col in columns}


def load_features(path: Path, feature_cols) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    lower_map = _lower_map(df.columns)
    resolved = []
    for col in feature_cols:
        key = str(col).strip().lower()
        if key not in lower_map:
            raise ValueError(
                "Missing columns in input file: " + ", ".join(feature_cols)
            )
        resolved.append(lower_map[key])
    return df[resolved]


def predict(args):
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_cols = payload.get("features", DEFAULT_FEATURES)
    else:
        model = payload
        feature_cols = DEFAULT_FEATURES

    X = load_features(input_path, feature_cols)

    preds = model.predict(X)
    result = X.copy()
    result["predicted_removal"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict removal using trained model")
    parser.add_argument("--model", default="models/removal_model.joblib")
    parser.add_argument("--input", required=True, help="CSV with feature columns")
    parser.add_argument(
        "--output",
        default="predictions/predictions.csv",
        help="Where to write predictions",
    )
    predict(parser.parse_args())
