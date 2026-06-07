#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
CONFIG="$1"
MODEL="$2"
EXP_ROOT="$3"      # e.g. experimentsCifar100
BASE_EXP="$4"      # e.g. teacher_wrn40_2
shift 4
PY_ARGS=("$@")     # Capture remaining args safely

# ------------------------
# Random free port (avoid conflicts)
# ------------------------
PORT=$((20000 + RANDOM % 20000))
export MASTER_PORT=$PORT

# ------------------------
# Environment
# ------------------------
export PYTHONPATH="$(pwd):$PYTHONPATH"

# ------------------------
# Training
# ------------------------
torchrun \
  --standalone \
  --nproc_per_node=1 \
  tools/train.py \
  -c "$CONFIG" \
  --model "$MODEL" \
  --experiment_root "$EXP_ROOT" \
  --experiment "$BASE_EXP" \
  "${PY_ARGS[@]}"

# ------------------------
# Plot losses
# ------------------------
python tools/plot_losses.py \
  --exp-dir "${EXP_ROOT}/${BASE_EXP}" \
  --out "loss_curves.png" \
  --title "${BASE_EXP}"

python tools/collect_seed_results.py \
  --exp-dir "${EXP_ROOT}/${BASE_EXP}" \
  --experiment "${BASE_EXP}" \
  --monitor "top1" \
  --mode "max" \
  --csv "${EXP_ROOT}/${BASE_EXP}/summary.csv"


