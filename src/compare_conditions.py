"""
Script to compare predictions across different temperature and pH conditions
and visualize the differences.
"""

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def generate_comparison_data(output_path: Path) -> pd.DataFrame:
    """Generate data with multiple temperature and pH combinations."""
    times = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Generate more granular absorbances for better resolution
    absorbances = np.linspace(0.01, 0.05, 10)
    
    # Generate sorted concentration values (ensuring proper ordering)
    concentrations = np.linspace(50, 300, 15)
    concentrations = np.sort(concentrations)  # Ensure sorted ascending order
    
    amounts_adsorbed = np.linspace(7, 9, 10)
    temperatures = np.array([15, 20, 25, 30, 35, 40])
    ph_values = np.array([5, 6, 7, 8, 9])

    data = []
    for time in times:
        for absorbance in absorbances:
            for conc in concentrations:
                for amount in amounts_adsorbed:
                    for temp in temperatures:
                        for ph in ph_values:
                            data.append({
                                "Time (min)": time,
                                "Absorbance": round(absorbance, 4),
                                "Concentration (mg/l)": round(conc, 2),
                                "Amount adsorbed (mg/g)": round(amount, 2),
                                "temperature": temp,
                                "PH": ph,
                            })

    df = pd.DataFrame(data)
    
    # Sort to ensure concentration ordering
    df = df.sort_values(by=['Time (min)', 'Absorbance', 'Concentration (mg/l)'])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} comparison records")
    
    return df


def predict_batch(model_path: Path, input_path: Path) -> pd.DataFrame:
    """Make predictions using the trained model."""
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        model = payload["model"]
        feature_cols = payload.get("features", DEFAULT_FEATURES)
    else:
        model = payload
        feature_cols = DEFAULT_FEATURES

    # Load full dataframe first
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        full_df = pd.read_excel(input_path)
    else:
        full_df = pd.read_csv(input_path)
    
    # Extract features for prediction
    X = load_features(input_path, feature_cols)
    preds = model.predict(X)
    
    # Combine all columns from input with predictions
    result = full_df.copy()
    result["predicted_removal"] = preds
    
    return result


def create_comparison_visualizations(predictions_df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations across conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Heatmap: Temperature vs Removal (at fixed concentrations)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Effect of Temperature on Removal % (pH=7, varying Concentrations)", fontsize=16)
    
    concentrations = sorted(predictions_df["Concentration (mg/l)"].unique())
    # Sample 6 evenly-spaced concentrations for visualization
    step = max(1, len(concentrations) // 6)
    sampled_concentrations = concentrations[::step][:6]
    
    temp_removal = predictions_df[predictions_df["PH"] == 7]
    
    for idx, conc in enumerate(sampled_concentrations):
        ax = axes[idx // 3, idx % 3]
        subset = temp_removal[temp_removal["Concentration (mg/l)"] == conc]
        
        pivot = subset.pivot_table(
            values="predicted_removal",
            index="Time (min)",
            columns="temperature",
            aggfunc="mean"
        )
        
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax, 
                   cbar_kws={"label": "Removal %"}, vmin=0, vmax=100)
        ax.set_title(f"Concentration: {conc:.0f} mg/l")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Time (min)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_effect_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: temperature_effect_heatmap.png")
    plt.close()

    # 2. Heatmap: pH vs Removal (at fixed concentrations)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Effect of pH on Removal % (T=25°C, varying Concentrations)", fontsize=16)
    
    temp_25 = predictions_df[predictions_df["temperature"] == 25]
    
    for idx, conc in enumerate(sampled_concentrations):
        ax = axes[idx // 3, idx % 3]
        subset = temp_25[temp_25["Concentration (mg/l)"] == conc]
        
        pivot = subset.pivot_table(
            values="predicted_removal",
            index="Time (min)",
            columns="PH",
            aggfunc="mean"
        )
        
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
                   cbar_kws={"label": "Removal %"}, vmin=0, vmax=100)
        ax.set_title(f"Concentration: {conc:.0f} mg/l")
        ax.set_xlabel("pH")
        ax.set_ylabel("Time (min)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "ph_effect_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: ph_effect_heatmap.png")
    plt.close()

    # 3. Line plot: Temperature effect on removal
    plt.figure(figsize=(14, 7))
    
    # Use middle concentration from sampled set
    mid_conc = sampled_concentrations[len(sampled_concentrations)//2]
    subset = predictions_df[(predictions_df["PH"] == 7) & 
                           (predictions_df["Concentration (mg/l)"] == mid_conc)]
    
    for time in [10, 30, 50, 70, 90]:
        time_data = subset[subset["Time (min)"] == time]
        grouped = time_data.groupby("temperature")["predicted_removal"].mean()
        plt.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f"Time: {time} min")
    
    plt.xlabel("Temperature (°C)", fontsize=12)
    plt.ylabel("Predicted Removal (%)", fontsize=12)
    plt.title(f"Temperature Effect on Removal (pH=7, Conc={mid_conc:.0f} mg/l)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_effect_lines.png", dpi=300, bbox_inches='tight')
    print(f"Saved: temperature_effect_lines.png")
    plt.close()

    # 4. Line plot: pH effect on removal
    plt.figure(figsize=(14, 7))
    
    subset = predictions_df[(predictions_df["temperature"] == 25) & 
                           (predictions_df["Concentration (mg/l)"] == mid_conc)]
    
    for time in [10, 30, 50, 70, 90]:
        time_data = subset[subset["Time (min)"] == time]
        grouped = time_data.groupby("PH")["predicted_removal"].mean()
        plt.plot(grouped.index, grouped.values, marker='s', linewidth=2, label=f"Time: {time} min")
    
    plt.xlabel("pH", fontsize=12)
    plt.ylabel("Predicted Removal (%)", fontsize=12)
    plt.title(f"pH Effect on Removal (T=25°C, Conc={mid_conc:.0f} mg/l)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "ph_effect_lines.png", dpi=300, bbox_inches='tight')
    print(f"Saved: ph_effect_lines.png")
    plt.close()

    # 5. 3D comparison: Temperature vs pH vs Removal
    fig = plt.figure(figsize=(16, 6))
    
    # Subplot 1: Fixed time, varying temp and pH
    ax1 = fig.add_subplot(121, projection='3d')
    
    subset = predictions_df[(predictions_df["Time (min)"] == 50) & 
                           (predictions_df["Concentration (mg/l)"] == mid_conc)]
    
    temps = sorted(subset["temperature"].unique())
    phs = sorted(subset["PH"].unique())
    
    X = np.zeros((len(temps), len(phs)))
    Y = np.zeros((len(temps), len(phs)))
    Z = np.zeros((len(temps), len(phs)))
    
    for i, temp in enumerate(temps):
        for j, ph in enumerate(phs):
            X[i, j] = temp
            Y[i, j] = ph
            z_vals = subset[(subset["temperature"] == temp) & (subset["PH"] == ph)]["predicted_removal"]
            Z[i, j] = z_vals.mean() if len(z_vals) > 0 else 0
    
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("pH")
    ax1.set_zlabel("Removal (%)")
    ax1.set_title(f"Temperature & pH Effect (Time=50 min, Conc={mid_conc:.0f} mg/l)")
    fig.colorbar(surf1, ax=ax1, label='Removal (%)')
    
    # Subplot 2: Fixed temperature and pH, varying time
    ax2 = fig.add_subplot(122, projection='3d')
    
    subset2 = predictions_df[(predictions_df["temperature"] == 25) & 
                            (predictions_df["PH"] == 7)]
    
    times = sorted(subset2["Time (min)"].unique())
    concs = sorted(subset2["Concentration (mg/l)"].unique())
    
    X2 = np.zeros((len(times), len(concs)))
    Y2 = np.zeros((len(times), len(concs)))
    Z2 = np.zeros((len(times), len(concs)))
    
    for i, time in enumerate(times):
        for j, conc in enumerate(concs):
            X2[i, j] = time
            Y2[i, j] = conc
            z_vals = subset2[(subset2["Time (min)"] == time) & (subset2["Concentration (mg/l)"] == conc)]["predicted_removal"]
            Z2[i, j] = z_vals.mean() if len(z_vals) > 0 else 0
    
    surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='plasma', alpha=0.8)
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Concentration (mg/l)")
    ax2.set_zlabel("Removal (%)")
    ax2.set_title("Time & Concentration Effect (T=25°C, pH=7)")
    fig.colorbar(surf2, ax=ax2, label='Removal (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "3d_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: 3d_comparison.png")
    plt.close()

    # 6. Difference from baseline (T=25°C, pH=7) heatmap
    baseline = predictions_df[(predictions_df["temperature"] == 25) & 
                             (predictions_df["PH"] == 7)].copy()
    baseline = baseline.groupby(["Time (min)", "Concentration (mg/l)"])["predicted_removal"].mean().reset_index()
    baseline = baseline.rename(columns={"predicted_removal": "baseline_removal"})
    
    # Compare other temperatures
    temp_comparison = predictions_df[predictions_df["PH"] == 7].copy()
    temp_comparison = temp_comparison.merge(
        baseline, 
        on=["Time (min)", "Concentration (mg/l)"],
        how="left"
    )
    temp_comparison["difference"] = temp_comparison["predicted_removal"] - temp_comparison["baseline_removal"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Difference from Baseline (T=25°C, pH=7) at Different Temperatures", fontsize=14)
    
    temperatures_to_plot = [15, 20, 30, 35, 40]
    
    for idx, temp in enumerate(temperatures_to_plot):
        ax = axes[idx]
        subset = temp_comparison[temp_comparison["temperature"] == temp]
        
        pivot = subset.pivot_table(
            values="difference",
            index="Time (min)",
            columns="Concentration (mg/l)",
            aggfunc="mean"
        )
        
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdBu_r", ax=ax,
                   cbar_kws={"label": "Difference (%)"}, center=0, vmin=-20, vmax=20)
        ax.set_title(f"T={temp}°C vs T=25°C")
        ax.set_xlabel("Concentration (mg/l)")
        ax.set_ylabel("Time (min)" if idx == 0 else "")
    
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_difference_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: temperature_difference_heatmap.png")
    plt.close()

    # 7. Difference from baseline for different pH values
    baseline_ph = predictions_df[(predictions_df["temperature"] == 25) & 
                                (predictions_df["PH"] == 7)].copy()
    baseline_ph = baseline_ph.groupby(["Time (min)", "Concentration (mg/l)"])["predicted_removal"].mean().reset_index()
    baseline_ph = baseline_ph.rename(columns={"predicted_removal": "baseline_removal"})
    
    ph_comparison = predictions_df[predictions_df["temperature"] == 25].copy()
    ph_comparison = ph_comparison.merge(
        baseline_ph,
        on=["Time (min)", "Concentration (mg/l)"],
        how="left"
    )
    ph_comparison["difference"] = ph_comparison["predicted_removal"] - ph_comparison["baseline_removal"]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Difference from Baseline (T=25°C, pH=7) at Different pH Values", fontsize=14)
    
    ph_to_plot = [5, 6, 8, 9]
    
    for idx, ph in enumerate(ph_to_plot):
        ax = axes[idx]
        subset = ph_comparison[ph_comparison["PH"] == ph]
        
        pivot = subset.pivot_table(
            values="difference",
            index="Time (min)",
            columns="Concentration (mg/l)",
            aggfunc="mean"
        )
        
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdBu_r", ax=ax,
                   cbar_kws={"label": "Difference (%)"}, center=0, vmin=-20, vmax=20)
        ax.set_title(f"pH={ph} vs pH=7")
        ax.set_xlabel("Concentration (mg/l)")
        ax.set_ylabel("Time (min)" if idx == 0 else "")
    
    plt.tight_layout()
    plt.savefig(output_dir / "ph_difference_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: ph_difference_heatmap.png")
    plt.close()

    print(f"\nAll comparison visualizations saved to: {output_dir}")


def main(args):
    """Main execution function."""
    model_path = Path(args.model)
    comparison_data_path = Path(args.comparison_data)
    output_dir = Path(args.output)

    print("=" * 70)
    print("Condition Comparison Analysis")
    print("=" * 70)

    # Generate comparison data
    print("\nGenerating comparison data across all temperature and pH combinations...")
    generate_comparison_data(comparison_data_path)

    # Make predictions
    print(f"\nLoading model from: {model_path}")
    print("Making predictions across all conditions...")
    predictions_df = predict_batch(model_path, comparison_data_path)
    
    # Save predictions
    output_path = output_dir / "comparison_predictions.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    # Create visualizations
    print(f"\nCreating comparison visualizations...")
    create_comparison_visualizations(predictions_df, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    for temp in sorted(predictions_df["temperature"].unique()):
        for ph in sorted(predictions_df["PH"].unique()):
            subset = predictions_df[(predictions_df["temperature"] == temp) & 
                                   (predictions_df["PH"] == ph)]
            mean_removal = subset["predicted_removal"].mean()
            print(f"T={temp:2.0f}°C, pH={ph}: Mean Removal = {mean_removal:.2f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare predictions across different temperatures and pH values"
    )
    parser.add_argument(
        "--model",
        default="models/removal_model.joblib",
        help="Path to trained model"
    )
    parser.add_argument(
        "--comparison-data",
        default="data/comparison_data.csv",
        help="Path to comparison data CSV (will be generated if not exists)"
    )
    parser.add_argument(
        "--output",
        default="reports/comparison",
        help="Output directory for comparison visualizations"
    )

    args = parser.parse_args()
    main(args)
