#!/bin/bash
set -e

# ============================
# Usage:
# bash plot_all_seeds.sh experimentsCifar10 gvd
# ============================

EXP_ROOT=$1      # e.g. experimentsCifar10
BASE_EXP=$2      # e.g. ged

if [[ -z "$EXP_ROOT" || -z "$BASE_EXP" ]]; then
  echo "Usage: bash plot_all_seeds.sh <EXP_ROOT> <BASE_EXP>"
  echo "Example: bash plot_all_seeds.sh experimentsCifar10 ged"
  exit 1
fi

BASE_DIR="${EXP_ROOT}/${BASE_EXP}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "ERROR: Directory not found: $BASE_DIR"
  exit 2
fi

echo "======================================"
echo "Plotting losses under: $BASE_DIR"
echo "======================================"

# ----------------------------------
# Plot each seed separately
# ----------------------------------
for SEED_DIR in "${BASE_DIR}"/seed*; do
  if [[ -d "$SEED_DIR" ]]; then

    SEED_NAME=$(basename "$SEED_DIR")

    echo "→ Plotting $SEED_NAME"

    python tools/plot_losses.py \
      --exp-dir "$SEED_DIR" \
      --out "loss_curves.png" \
      --title "${BASE_EXP}/${SEED_NAME}"

  fi
done


echo "All plots saved."



#bash tools/plot_losses.sh exp_cifar10 fits