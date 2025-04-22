import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract error rates as a single-row table.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--out_csv", type=str, default="performance_row.csv", help="Path to save the output CSV")
    args = parser.parse_args()

    try:
        # Read input CSV file
        df = pd.read_csv(args.csv_path)
        print("✅ Loaded:", args.csv_path)

        # Filter out rows with valid round info (e.g., containing '_rep')
        df_valid = df[df["Corruption"].str.contains("_rep")].copy()

        # Extract round number from 'Corruption' column
        df_valid["Round"] = df_valid["Corruption"].apply(lambda x: int(x.split("_rep")[-1]))

        # Compute average error rate for each round
        round_error = df_valid.groupby("Round")["Error Rate"].mean()

        # Convert to dictionary and append overall average
        row_data = round_error.to_dict()
        row_data["Avg"] = round_error.mean()

        # Create a one-row DataFrame and save
        df_out = pd.DataFrame([row_data])
        df_out.to_csv(args.out_csv, index=False)
        print(f"✅ Output saved to: {args.out_csv}")

    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    main()
