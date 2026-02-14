"""
Script to generate predictions and create visualizations for removal percentage.
This script handles new data generation, predictions, and creates graphs showing
the relationship between removal percentage, time, and adsorbent concentration.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    """Load features from CSV or Excel file."""
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


def generate_sample_data(output_path: Path, temperature: float = 25, ph: float = 7) -> pd.DataFrame:
    """Generate sample new data for prediction with fixed temperature and pH."""
    # Create combinations of different parameter values
    # These match the trained model's expected features plus temperature and pH
    times = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Generate dosage values with 4 decimal places for higher precision
    dosages = np.linspace(0.2000, 1.0000, 20)  # 20 values from 0.2 to 1.0
    
    # Absorbances corresponding to dosages
    absorbances = dosages / 20  # Inverse calculation to get absorbance
    
    # Generate sorted concentration values (ensuring each value <= next value)
    concentrations = np.linspace(50, 300, 15)  # More granular concentration steps
    concentrations = np.sort(concentrations)  # Ensure sorted order
    
    amounts_adsorbed = np.linspace(7, 9, 10)

    # Generate a grid of combinations with fixed temperature and pH
    data = []
    for time in times:
        for dosage, absorbance in zip(dosages, absorbances):
            for conc in concentrations:
                for amount in amounts_adsorbed:
                    data.append({
                        "time": time,
                        "dosage of adsorbent": round(dosage, 4),  # 4 decimal places
                        "temperature": temperature,
                        "PH": ph,
                        "Adsorbent Concentration": round(conc, 2),
                        # Also include ALT feature names for model compatibility
                        "Time (min)": time,
                        "Absorbance": round(absorbance, 4),
                        "Concentration (mg/l)": round(conc, 2),
                        "Amount adsorbed (mg/g)": round(amount, 2),
                    })

    df = pd.DataFrame(data)
    
    # Verify concentration ordering within each group
    df = df.sort_values(by=['time', 'dosage of adsorbent', 'Adsorbent Concentration'])
    
    # Format dosage column to always show 4 decimal places
    df['dosage of adsorbent'] = df['dosage of adsorbent'].apply(lambda x: f"{x:.4f}")
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Convert back for statistics (after saving)
    df['dosage of adsorbent'] = df['dosage of adsorbent'].astype(float)
    
    print(f"Generated {len(df)} sample records and saved to: {output_path}")
    print(f"Dosage range: {df['dosage of adsorbent'].min():.4f} to {df['dosage of adsorbent'].max():.4f}")
    print(f"Concentration range: {df['Adsorbent Concentration'].min():.2f} to {df['Adsorbent Concentration'].max():.2f}")
    
    return df


def predict(model_path: Path, input_path: Path, output_path: Path):
    """Make predictions using the trained model."""
    # Load model and features
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_cols = payload.get("features", DEFAULT_FEATURES)
    else:
        model = payload
        feature_cols = DEFAULT_FEATURES

    # Load ALL columns from input file
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        full_df = pd.read_excel(input_path)
    else:
        full_df = pd.read_csv(input_path)
    
    # Load only feature columns for prediction
    X = load_features(input_path, feature_cols)

    # Make predictions
    preds = model.predict(X)
    
    # Combine all input columns with predictions
    result = full_df.copy()
    result["predicted_removal"] = preds

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
    return result


def create_visualizations(predictions_df: pd.DataFrame, output_dir: Path, temperature: float = 25, ph: float = 7):
    """Create visualizations of the predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    title_suffix = f" (T={temperature}°C, pH={ph})"

    # 1. 2D Plot: Removal vs Time (colored by Concentration)
    plt.figure(figsize=(12, 6))
    
    # Get unique concentrations
    unique_conc = sorted(predictions_df["Concentration (mg/l)"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_conc)))
    
    for i, conc in enumerate(unique_conc[::max(1, len(unique_conc)//5)]):  # Sample every Nth concentration
        subset = predictions_df[predictions_df["Concentration (mg/l)"] == conc]
        # Group by time and get mean removal
        grouped = subset.groupby("Time (min)")["predicted_removal"].mean().reset_index()
        plt.plot(grouped["Time (min)"], grouped["predicted_removal"], 
                marker='o', label=f"Conc: {conc:.1f} mg/l", linewidth=2)
    
    plt.xlabel("Time (min)", fontsize=12)
    plt.ylabel("Predicted Removal (%)", fontsize=12)
    plt.title("Removal Percentage vs Time at Different Concentrations" + title_suffix, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "removal_vs_time_by_concentration.png", dpi=300, bbox_inches='tight')
    print(f"Saved graph: {output_dir / 'removal_vs_time_by_concentration.png'}")
    plt.close()

    # 2. 2D Plot: Removal vs Concentration (colored by Time)
    plt.figure(figsize=(12, 6))
    
    unique_times = sorted(predictions_df["Time (min)"].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(unique_times)))
    
    for i, t in enumerate(unique_times[::max(1, len(unique_times)//5)]):  # Sample every Nth time
        subset = predictions_df[predictions_df["Time (min)"] == t]
        # Group by concentration and get mean removal
        grouped = subset.groupby("Concentration (mg/l)")["predicted_removal"].mean().reset_index()
        plt.plot(grouped["Concentration (mg/l)"], grouped["predicted_removal"], 
                marker='s', label=f"Time: {t} min", color=colors[i], linewidth=2)
    
    plt.xlabel("Concentration (mg/l)", fontsize=12)
    plt.ylabel("Predicted Removal (%)", fontsize=12)
    plt.title("Removal Percentage vs Concentration at Different Times" + title_suffix, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "removal_vs_concentration_by_time.png", dpi=300, bbox_inches='tight')
    print(f"Saved graph: {output_dir / 'removal_vs_concentration_by_time.png'}")
    plt.close()

    # 3. Heatmap: Time vs Concentration (with Removal as color)
    plt.figure(figsize=(12, 8))
    
    # Create pivot table
    pivot = predictions_df.pivot_table(
        values="predicted_removal",
        index="Time (min)",
        columns="Concentration (mg/l)",
        aggfunc="mean"
    )
    
    im = plt.imshow(pivot.values, aspect='auto', cmap='RdYlGn', origin='lower')
    plt.colorbar(im, label='Predicted Removal (%)')
    
    plt.xlabel("Concentration (mg/l)", fontsize=12)
    plt.ylabel("Time (min)", fontsize=12)
    plt.title("Removal Percentage Heatmap (Time vs Concentration)" + title_suffix, fontsize=14)
    
    # Set ticks - sample for readability
    x_ticks = np.linspace(0, len(pivot.columns)-1, min(10, len(pivot.columns)), dtype=int)
    y_ticks = np.linspace(0, len(pivot.index)-1, min(10, len(pivot.index)), dtype=int)
    
    plt.xticks(x_ticks, [f"{pivot.columns[i]:.0f}" for i in x_ticks], rotation=45)
    plt.yticks(y_ticks, [f"{pivot.index[i]:.0f}" for i in y_ticks])
    
    plt.tight_layout()
    plt.savefig(output_dir / "removal_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved graph: {output_dir / 'removal_heatmap.png'}")
    plt.close()

    # 4. 3D Surface Plot: Time vs Concentration vs Removal
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        times_unique = sorted(predictions_df["Time (min)"].unique())
        conc_unique = sorted(predictions_df["Concentration (mg/l)"].unique())
        
        X = np.zeros((len(times_unique), len(conc_unique)))
        Y = np.zeros((len(times_unique), len(conc_unique)))
        Z = np.zeros((len(times_unique), len(conc_unique)))
        
        for i, t in enumerate(times_unique):
            for j, c in enumerate(conc_unique):
                X[i, j] = t
                Y[i, j] = c
                z_vals = predictions_df[
                    (predictions_df["Time (min)"] == t) & 
                    (predictions_df["Concentration (mg/l)"] == c)
                ]["predicted_removal"]
                Z[i, j] = z_vals.mean() if len(z_vals) > 0 else 0
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel("Time (min)", fontsize=11)
        ax.set_ylabel("Concentration (mg/l)", fontsize=11)
        ax.set_zlabel("Predicted Removal (%)", fontsize=11)
        ax.set_title("3D Surface: Removal vs Time & Concentration" + title_suffix, fontsize=14)
        
        fig.colorbar(surf, ax=ax, label='Removal (%)')
        plt.savefig(output_dir / "removal_3d_surface.png", dpi=300, bbox_inches='tight')
        print(f"Saved graph: {output_dir / 'removal_3d_surface.png'}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create 3D surface plot: {e}")

    print(f"\nAll visualizations saved to: {output_dir}")


def main(args):
    """Main execution function."""
    model_path = Path(args.model)
    new_data_path = Path(args.new_data)
    predictions_output = Path(args.predictions_output)
    graphs_output = Path(args.graphs_output)
    temperature = float(args.temperature)
    ph = float(args.ph)

    print("=" * 60)
    print("Adsorption Removal Prediction and Visualization")
    print("=" * 60)

    # Generate sample data if new data file doesn't exist
    if not new_data_path.exists():
        print(f"\nGenerating sample data with Temperature={temperature}°C and pH={ph}...")
        generate_sample_data(new_data_path, temperature=temperature, ph=ph)
    else:
        print(f"\nUsing existing data from: {new_data_path}")

    # Make predictions
    print(f"\nLoading model from: {model_path}")
    print("Making predictions...")
    predictions_df = predict(model_path, new_data_path, predictions_output)

    # Add temperature and pH metadata to predictions
    predictions_df["temperature"] = temperature
    predictions_df["PH"] = ph
    
    # Format dosage column to show 4 decimal places if it exists
    if "dosage of adsorbent" in predictions_df.columns:
        # Save with formatted dosage
        dosage_col = predictions_df["dosage of adsorbent"]
        predictions_df["dosage of adsorbent"] = dosage_col.apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else x)
    
    predictions_df.to_csv(predictions_output, index=False)
    print(f"Predictions with temperature/pH metadata saved to: {predictions_output}")

    # Create visualizations
    print(f"\nCreating visualizations...")
    create_visualizations(predictions_df, graphs_output, temperature, ph)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Prediction Summary Statistics")
    print("=" * 60)
    print(f"Conditions: Temperature = {temperature}°C, pH = {ph}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"Mean removal: {predictions_df['predicted_removal'].mean():.2f}%")
    print(f"Min removal: {predictions_df['predicted_removal'].min():.2f}%")
    print(f"Max removal: {predictions_df['predicted_removal'].max():.2f}%")
    print(f"Std removal: {predictions_df['predicted_removal'].std():.2f}%")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions and visualizations for adsorption removal"
    )
    parser.add_argument(
        "--model",
        default="models/removal_model.joblib",
        help="Path to trained model"
    )
    parser.add_argument(
        "--new-data",
        default="data/new_data_for_prediction.csv",
        help="Path to new data CSV (will be generated if not exists)"
    )
    parser.add_argument(
        "--predictions-output",
        default="predictions/new_predictions.csv",
        help="Output path for predictions CSV"
    )
    parser.add_argument(
        "--graphs-output",
        default="reports/visualizations",
        help="Output directory for visualization graphs"
    )
    parser.add_argument(
        "--temperature",
        default=25.0,
        type=float,
        help="Fixed temperature in Celsius for predictions (default: 25)"
    )
    parser.add_argument(
        "--ph",
        default=7.0,
        type=float,
        help="Fixed pH value for predictions (default: 7)"
    )

    args = parser.parse_args()
    main(args)
