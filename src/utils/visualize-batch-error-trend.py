import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_error_trend_comparison(csv1, csv2, output_path="error_trend_comparison.png"):
    # Load CSV files
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Validate columns
    for df, name in zip([df1, df2], [csv1, csv2]):
        if "Batch Index" not in df.columns or "Error Rate" not in df.columns:
            raise ValueError(f"Missing expected columns in {name}")

    # Extract values
    x1, y1 = df1["Batch Index"], df1["Error Rate"]
    x2, y2 = df2["Batch Index"], df2["Error Rate"]
    avg1, avg2 = y1.mean(), y2.mean()

    # Plot
    plt.figure(figsize=(10, 4))
    label1 = os.path.basename(csv1)
    label2 = os.path.basename(csv2)

    plt.plot(x1, y1, label=label1, color='tab:blue', linewidth=1.5)
    plt.plot(x2, y2, label=label2, color='tab:orange', linewidth=1.5)

    # Draw average lines (dashed, same color)
    plt.axhline(avg1, color='tab:blue', linestyle='--', linewidth=1.2)
    plt.axhline(avg2, color='tab:orange', linestyle='--', linewidth=1.2)

    # Plot aesthetics
    plt.xlabel("Batch Index", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.title("TTA Error Rate Comparison", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    print(f"[✓] Saved comparison plot to: {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two TTA error trend CSV files with average lines.")
    parser.add_argument("csv1", type=str, help="Path to the first CSV file")
    parser.add_argument("csv2", type=str, help="Path to the second CSV file")
    parser.add_argument("--output", type=str, default="error_trend_comparison.png", help="Output figure path")
    args = parser.parse_args()

    plot_error_trend_comparison(args.csv1, args.csv2, args.output)
