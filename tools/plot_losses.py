#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def find_metrics_csv(exp_dir: str) -> str:
    # Common locations (adjust if your project differs)
    candidates = [
        os.path.join(exp_dir, "metrics.csv"),
        os.path.join(exp_dir, "checkpoints", "metrics.csv"),
        os.path.join(exp_dir, "logs", "metrics.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="Full path to experiment directory")
    ap.add_argument("--out", default="loss_curves.png", help="Output filename (saved under exp-dir unless absolute)")
    ap.add_argument("--title", default=None, help="Optional plot title")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    exp_dir = args.exp_dir
    if not os.path.isdir(exp_dir):
        print(f"[plot_losses] ERROR: exp-dir not found: {exp_dir}", file=sys.stderr)
        sys.exit(1)

    metrics_path = find_metrics_csv(exp_dir)
    if not metrics_path:
        print(
            f"[plot_losses] ERROR: metrics.csv not found under {exp_dir}\n"
            f"Checked: metrics.csv, checkpoints/metrics.csv, logs/metrics.csv",
            file=sys.stderr,
        )
        sys.exit(2)

    df = pd.read_csv(metrics_path)

    # Flexible column handling
    # Your CheckpointManager wrote: epoch,train_loss,test_loss,kd_loss,top1,top5
    required = ["epoch", "train_loss", "test_loss", "kd_loss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[plot_losses] ERROR: missing columns in {metrics_path}: {missing}", file=sys.stderr)
        print(f"[plot_losses] Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    df = df.sort_values("epoch")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(exp_dir, out_path)

    plt.figure()
    plt.plot(df["epoch"].to_numpy(), df["train_loss"].to_numpy(), label="train_loss")
    plt.plot(df["epoch"].to_numpy(), df["kd_loss"].to_numpy(), label="kd_loss")
    plt.plot(df["epoch"].to_numpy(), df["test_loss"].to_numpy(), label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, which="both")
    plt.legend()

    if args.title is None:
        # title defaults to the last 2 folder names for readability
        tail = os.path.normpath(exp_dir).split(os.sep)
        args.title = "/".join(tail[-2:]) if len(tail) >= 2 else exp_dir
    plt.title(args.title)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"[plot_losses] Saved: {out_path}")
    print(f"[plot_losses] Read : {metrics_path}")


if __name__ == "__main__":
    main()