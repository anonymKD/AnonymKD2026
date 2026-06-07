#!/bin/bash
set -e

CONFIG="$1"
MODEL="$2"
EXP_ROOT="$3"      # e.g. experimentsCifar100
BASE_EXP="$4"      # e.g. GVD
shift 4
PY_ARGS=("$@")     # capture the rest safely as an array

NUM_RUNS=5
GPUS=1

export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p logs "logs/${BASE_EXP}" "${EXP_ROOT}/${BASE_EXP}"

for i in $(seq 1 "$NUM_RUNS"); do
  PORT=$((20000 + RANDOM % 20000))
  EXP_NAME="${BASE_EXP}/seed${i}"

  echo "=== Run $i/${NUM_RUNS} | port=$PORT | exp=$EXP_NAME ==="

  MASTER_PORT=$PORT \
  torchrun --standalone --nproc_per_node="${GPUS}" \
    tools/train.py \
    -c "$CONFIG" \
    --model "$MODEL" \
    --experiment_root "$EXP_ROOT" \
    --experiment "$EXP_NAME" \
    --seed "$i" \
    "${PY_ARGS[@]}" \
    2>&1 | tee "logs/${EXP_NAME}.log"

  python tools/collect_seed_results.py \
    --exp-dir "${EXP_ROOT}/${EXP_NAME}" \
    --experiment "$EXP_NAME" \
    --seed "$i" \
    --monitor "top1" \
    --mode "max" \
    --csv "${EXP_ROOT}/${BASE_EXP}/all_seeds_summary.csv"

  SEED_NAME="seed${i}"
  echo "→ Plotting ${BASE_EXP}/${SEED_NAME}"
  python tools/plot_losses.py \
    --exp-dir "${EXP_ROOT}/${EXP_NAME}" \
    --out "../plots/loss_curves_${SEED_NAME}.png" \
    --title "${BASE_EXP}/${SEED_NAME}"
done

python tools/add_avg_row.py --csv "${EXP_ROOT}/${BASE_EXP}/all_seeds_summary.csv"

