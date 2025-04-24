import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Append round-wise error rates to performance.csv with method label.")
    parser.add_argument("csv_path", type=str, help="Path to input CSV")
    parser.add_argument("method", type=str, help="Name of the method (will be the first column)")
    parser.add_argument("--out_csv", type=str, default="performance.csv", help="Path to output CSV")
    args = parser.parse_args()

    # Load input CSV
    df = pd.read_csv(args.csv_path)

    # Extract round number and average error rate
    df_valid = df[df["Corruption"].str.contains("_rep")].copy()
    df_valid["Round"] = df_valid["Corruption"].apply(lambda x: int(x.split("_rep")[-1]))

    round_error = df_valid.groupby("Round")["Error Rate"].mean()
    round_dict = round_error.to_dict()
    round_dict["Avg"] = round_error.mean()

    # Prepare ordered columns
    sorted_keys = sorted(round_dict.keys(), key=lambda x: (x if isinstance(x, int) else float('inf')))
    column_names = ["Method"] + [str(k) for k in sorted_keys]
    values = [args.method] + [round(round_dict[k], 2) for k in sorted_keys]
    new_row = pd.DataFrame([values], columns=column_names)

    # Append to or create performance file
    if os.path.exists(args.out_csv):
        existing_df = pd.read_csv(args.out_csv)
        combined_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        combined_df = new_row

    # Save
    combined_df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
