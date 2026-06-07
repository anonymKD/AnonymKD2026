import argparse
import os
import re
import pandas as pd


def read_metrics_csv(exp_dir: str) -> pd.DataFrame:
    fp = os.path.join(exp_dir, "metrics.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"metrics.csv not found: {fp}")

    df = pd.read_csv(fp)

    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    return df


def pick_best_row(df: pd.DataFrame, monitor: str, mode: str) -> pd.Series:
    if monitor not in df.columns:
        raise ValueError(
            f"Metric '{monitor}' not in columns: {list(df.columns)}"
        )

    s = pd.to_numeric(df[monitor], errors="coerce")

    if mode == "max":
        idx = s.idxmax()
    else:
        idx = s.idxmin()

    return df.loc[idx]


def pick_last_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]


def infer_seed(exp: str):
    m = re.search(r"(?:seed)(\d+)", exp)
    return int(m.group(1)) if m else None


def append_csv(csv_path: str, row_df: pd.DataFrame):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if not os.path.exists(csv_path):
        # write with header
        row_df.to_csv(csv_path, index=False)
    else:
        # append without header
        row_df.to_csv(csv_path, mode="a", header=False, index=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--monitor", default="top1")
    ap.add_argument("--mode", choices=["max", "min"], default="max")
    ap.add_argument("--csv", required=True)

    args = ap.parse_args()

    df = read_metrics_csv(args.exp_dir)

    best = pick_best_row(df, args.monitor, args.mode)
    last = pick_last_row(df)

    seed = args.seed
    if seed is None:
        seed = infer_seed(args.experiment)

    out = {
        "experiment": args.experiment,
        "seed": seed,

        "best_epoch": int(best["epoch"]),
        "best_train_loss": float(best["train_loss"]),
        "best_test_loss": float(best["test_loss"]),
        "best_kd_loss": float(best["kd_loss"]),
        "best_top1": float(best["top1"]),
        "best_top5": float(best["top5"]),

        "last_epoch": int(last["epoch"]),
        "last_train_loss": float(last["train_loss"]),
        "last_test_loss": float(last["test_loss"]),
        "last_kd_loss": float(last["kd_loss"]),
        "last_top1": float(last["top1"]),
        "last_top5": float(last["top5"]),
    }

    row_df = pd.DataFrame([out])

    append_csv(args.csv, row_df)

    print(f"Appended results for {args.experiment} -> {args.csv}")


if __name__ == "__main__":
    main()