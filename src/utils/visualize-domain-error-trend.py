import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Visualize domain-wise error rates from multiple CSV files with average lines.")
    parser.add_argument("csv_paths", type=str, nargs="+", help="Paths to one or more CSV files")
    args = parser.parse_args()

    plt.figure(figsize=(12, 5))
    all_avgs = []

    for i, csv_path in enumerate(args.csv_paths):
        df = pd.read_csv(csv_path)

        if "Error Rate" not in df.columns:
            raise ValueError(f"[✗] 'Error Rate' column not found in {csv_path}")

        error_rates = df["Error Rate"].values
        domain_ids = list(range(len(error_rates)))
        avg = error_rates.mean()
        all_avgs.append(avg)

        label = os.path.basename(csv_path)
        color = f"tab:{['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'][i % 8]}"

        plt.plot(domain_ids, error_rates, linestyle='-', color=color, label=label)
        plt.axhline(avg, color=color, linestyle='--', linewidth=1.5)

    # Aesthetics
    plt.title("Domain-wise Error Rate Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Domain Index", fontsize=12)
    plt.ylabel("Error Rate", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Save figure
    output_dir = os.path.dirname(args.csv_paths[0])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"error_trend_compare_{len(args.csv_paths)}files.png")
    plt.savefig(save_path, dpi=150)
    print(f"[✓] Saved plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
