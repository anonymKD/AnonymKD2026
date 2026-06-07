import argparse
import pandas as pd
import numpy as np
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to all_seeds_summary.csv")
    ap.add_argument("--label", default="AVG", help="Value to put in experiment column")
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # If an AVG row already exists, remove it (so re-running won't duplicate)
    if "experiment" in df.columns:
        df = df[df["experiment"].astype(str) != args.label].copy()

    # numeric columns only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    avg_row = {col: "" for col in df.columns}
    if "experiment" in df.columns:
        avg_row["experiment"] = args.label
    if "dataset" in df.columns:
        avg_row["dataset"] = args.label
    if "seed" in df.columns:
        avg_row["seed"] = "avg"

    for c in num_cols:
        avg_row[c] = float(df[c].mean())

    df_out = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    df_out.to_csv(csv_path, index=False)

    print(f"Added AVG row to {csv_path}")

if __name__ == "__main__":
    main()