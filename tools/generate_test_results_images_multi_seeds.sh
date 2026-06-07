#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
KD="$1"
MODEL="$2"
shift 2
PY_ARGS=("$@")

#KD="mse"
# MODEL="cifar_resent8x4"
EXP_ROOT_BASE="exp_limited/0.4/"
CONFIG="configs/train_cifar100_resnet32x4_8x4.yaml"

NUM_RUNS=5
ENTRY="tools/test.py"
TEST_RESULT_FOLDER="all_test_results"

EXP_ROOT="${EXP_ROOT_BASE}/${MODEL}"

RESULT_DIR="${EXP_ROOT_BASE}/${MODEL}/${TEST_RESULT_FOLDER}"
mkdir -p "$RESULT_DIR"

# dataset-specific temporary csv containing 3 seed test rows
DATASET_PREFIX="${KD}"
SUMMARY_CSV_TEMP="${RESULT_DIR}/${DATASET_PREFIX}_summary.csv"


# start fresh for this dataset
rm -f "$SUMMARY_CSV_TEMP"

for i in $(seq 1 "$NUM_RUNS"); do
sh tools/dist_run.sh \
    "$ENTRY" \
    "$CONFIG" \
    "$MODEL" \
    --experiment_root "$EXP_ROOT" \
    --experiment "$TEST_RESULT_FOLDER" \
    --resume "${EXP_ROOT}/${KD}/seed${i}/best.pth.tar" \
    --prefix_filename "$DATASET_PREFIX" \
    "${PY_ARGS[@]}"
done

# add average row to this dataset-specific temp csv
python tools/add_avg_row.py --csv "$SUMMARY_CSV_TEMP"

# delete log files in result directory
rm -f "${RESULT_DIR}"/log_*





