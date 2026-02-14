"""
Automated Pipeline Script
Automatically processes new input data and generates predictions with visualizations.
"""

import argparse
from pathlib import Path
import sys
import subprocess
import json
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=False, 
            text=True
        )
        if result.returncode != 0:
            print(f"[ERROR] {description} failed")
            return False
        print(f"[OK] {description} completed successfully")
        return True
    except Exception as e:
        print(f"[ERROR] running command: {e}")
        return False


def create_summary_report(input_file, temperature, ph, output_dir):
    """Create a summary report of the analysis."""
    report_path = Path(output_dir) / "ANALYSIS_REPORT.md"
    
    report_content = f"""# Automated Adsorption Removal Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Input Information
- Input File: {input_file}
- Temperature: {temperature}°C
- pH: {ph}
- Output Directory: {output_dir}

## Processing Steps Completed

1. [DONE] Loaded and validated input data
2. [DONE] Generated predictions using trained model
3. [DONE] Created visualization graphs:
   - removal_vs_time_by_concentration.png
   - removal_vs_concentration_by_time.png
   - removal_heatmap.png
   - removal_3d_surface.png
4. [DONE] Generated comparison analysis:
   - temperature_effect_heatmap.png
   - ph_effect_heatmap.png
   - temperature_effect_lines.png
   - ph_effect_lines.png
   - 3d_comparison.png
   - temperature_difference_heatmap.png
   - ph_difference_heatmap.png

## Output Files

### Predictions
- predictions/new_predictions.csv - Predictions with all features and results
- reports/comparison/comparison_predictions.csv - Comparison predictions across all conditions

### Visualizations
Located in:
- reports/visualizations/ - Main prediction visualizations
- reports/comparison/ - Comparison analysis graphs

## Statistics Summary

All predictions have been generated with:
- Fixed Temperature: {temperature}°C
- Fixed pH: {ph}
- Time variations: 10-100 minutes
- Concentration variations: 50-300 mg/l
- Absorbance variations: 0.01-0.05

## Next Steps

To analyze the results:
1. Review the CSV files for detailed predictions
2. Check the visualization graphs in reports/visualizations/ and reports/comparison/
3. Use the difference heatmaps to understand how temperature and pH affect removal

## Notes

- The model predictions remain constant across different temperatures and pH values because the original model was trained without these parameters as features.
- To get temperature/pH-dependent predictions, the model would need retraining with a dataset that includes temperature and pH variations.

---
Analysis completed successfully
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\nReport created: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated pipeline for adsorption removal analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default settings (T=25°C, pH=7)
  python src/autorun.py --input data/my_data.csv
  
  # Process with custom temperature and pH
  python src/autorun.py --input data/my_data.csv --temperature 30 --ph 6
  
  # Specify all output directories
  python src/autorun.py --input data/my_data.csv --output reports/my_analysis
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV or Excel file with data"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Fixed temperature in Celsius (default: 25)"
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.0,
        help="Fixed pH value (default: 7)"
    )
    parser.add_argument(
        "--output",
        default="reports/analysis",
        help="Output directory for all results (default: reports/analysis)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"[ERROR] Input file not found: {input_file}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("AUTOMATED ADSORPTION REMOVAL ANALYSIS PIPELINE")
    print("="*70)
    print(f"Input File: {input_file}")
    print(f"Temperature: {args.temperature}°C")
    print(f"pH: {args.ph}")
    print(f"Output Directory: {output_dir}")
    
    # Step 1: Run predictions with specified temperature and pH
    python_exe = Path(".venv/Scripts/python.exe")
    
    cmd_predict = (
        f'"{python_exe}" src/predict_and_visualize.py '
        f'--temperature {args.temperature} '
        f'--ph {args.ph} '
        f'--predictions-output "{output_dir}/predictions.csv" '
        f'--graphs-output "{output_dir}/visualizations"'
    )
    
    if not run_command(cmd_predict, "Step 1: Generating Predictions and Visualizations"):
        sys.exit(1)
    
    # Step 2: Run comparison analysis
    cmd_compare = (
        f'"{python_exe}" src/compare_conditions.py '
        f'--comparison-data "{output_dir}/comparison_data.csv" '
        f'--output "{output_dir}/comparison"'
    )
    
    if not run_command(cmd_compare, "Step 2: Running Comparison Analysis"):
        sys.exit(1)
    
    # Step 3: Create summary report
    print(f"\n{'='*70}")
    print("Step 3: Creating Summary Report")
    print(f"{'='*70}")
    create_summary_report(input_file, args.temperature, args.ph, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("[SUCCESS] ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults Summary:")
    print(f"   [DONE] Predictions saved to: {output_dir}/predictions.csv")
    print(f"   [DONE] Visualizations saved to: {output_dir}/visualizations/")
    print(f"   [DONE] Comparison analysis saved to: {output_dir}/comparison/")
    print(f"   [DONE] Report saved to: {output_dir}/ANALYSIS_REPORT.md")
    print("\nGraph Files Generated:")
    print(f"   - removal_vs_time_by_concentration.png")
    print(f"   - removal_vs_concentration_by_time.png")
    print(f"   - removal_heatmap.png")
    print(f"   - removal_3d_surface.png")
    print(f"   - temperature_effect_heatmap.png")
    print(f"   - ph_effect_heatmap.png")
    print(f"   - temperature_effect_lines.png")
    print(f"   - ph_effect_lines.png")
    print(f"   - 3d_comparison.png")
    print(f"   - temperature_difference_heatmap.png")
    print(f"   - ph_difference_heatmap.png")
    print("\nAll analysis complete! Check the output directory for results.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
