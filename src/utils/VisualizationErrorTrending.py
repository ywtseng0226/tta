import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize domain-wise error rates from two CSV files with average lines (no legend for avg lines).")
    parser.add_argument("csv_path1", type=str, help="Path to first CSV file")
    parser.add_argument("csv_path2", type=str, help="Path to second CSV file")
    args = parser.parse_args()

    # Load CSVs
    df1 = pd.read_csv(args.csv_path1)
    df2 = pd.read_csv(args.csv_path2)

    if "Error Rate" not in df1.columns or "Error Rate" not in df2.columns:
        raise ValueError("Expected column 'Error Rate' not found in one or both CSV files.")

    # Extract values
    error_rates1 = df1["Error Rate"].values
    error_rates2 = df2["Error Rate"].values
    domain_ids = list(range(len(error_rates1)))

    avg1 = error_rates1.mean()
    avg2 = error_rates2.mean()

    # Plot
    plt.figure(figsize=(12, 5))
    label1 = os.path.basename(args.csv_path1)
    label2 = os.path.basename(args.csv_path2)

    plt.plot(domain_ids, error_rates1, linestyle='-', color='tab:blue', label=label1)
    plt.plot(domain_ids, error_rates2, linestyle='-', color='tab:orange', label=label2)

    plt.axhline(avg1, color='tab:blue', linestyle='--', linewidth=1.5)
    plt.axhline(avg2, color='tab:orange', linestyle='--', linewidth=1.5)

    # Aesthetics
    plt.title("Domain-wise Error Rate Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Domain Index", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Save figure
    output_dir = os.path.dirname(args.csv_path1)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "error_trend_compare_avg.png")
    plt.savefig(save_path, dpi=150)
    print(f"[✓] Saved plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
