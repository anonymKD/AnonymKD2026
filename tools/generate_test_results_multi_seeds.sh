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

# MODEL="ts_lstm_32_2"
EXP_ROOT_BASE="exp_ts_students_ms"
NUM_RUNS=5

ENTRY="tools/test.py"
CONFIG="configs/test_ts_student.yaml"
TEST_RESULT_FOLDER="all_test_results"

EXP_ROOT="${EXP_ROOT_BASE}/${MODEL}"

DATASETS=(
  Computers
  UWaveGestureLibraryAll
  Strawberry
  BeetleFly
  wafer
  Lighting2
  ItalyPowerDemand
  yoga
  Trace
  ShapesAll
  MiddlePhalanxOutlineCorrect
  SwedishLeaf
  FaceAll
  StarLightCurves
  ECG200
  MoteStrain
  SonyAIBORobotSurfaceII
  Ham
  NonInvasiveFatalECG_Thorax1
  NonInvasiveFatalECG_Thorax2
)

RESULT_DIR="${EXP_ROOT_BASE}/${MODEL}/${TEST_RESULT_FOLDER}"
mkdir -p "$RESULT_DIR"

# final dataset-level average summary
SUMMARY_CSV="${RESULT_DIR}/${KD}_summary_avg.csv"
> "$SUMMARY_CSV"
FIRST=1

for DATASET in "${DATASETS[@]}"; do
  # dataset-specific temporary csv containing 3 seed test rows
  DATASET_PREFIX="${KD}_${DATASET}"
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
      --dataset "$DATASET" \
      --resume "${EXP_ROOT}/${KD}/${DATASET}/seed${i}/last.pth.tar" \
      --prefix_filename "$DATASET_PREFIX" \
      "${PY_ARGS[@]}"
  done

  # add average row to this dataset-specific temp csv
  python tools/add_avg_row.py --csv "$SUMMARY_CSV_TEMP"

  # initialize final summary header once
  if [ "$FIRST" -eq 1 ]; then
    HEADER=$(head -n 1 "$SUMMARY_CSV_TEMP")
    echo "dataset,${HEADER}" > "$SUMMARY_CSV"
    FIRST=0
  fi

  # append only the AVERAGE row from this dataset temp csv
  AVG_LINE=$(awk -F',' '$1=="AVG" {print; exit}' "$SUMMARY_CSV_TEMP")
  if [ -n "$AVG_LINE" ]; then
    echo "${DATASET},${AVG_LINE}" >> "$SUMMARY_CSV"
  else
    echo "Warning: no AVERAGE row found in $SUMMARY_CSV_TEMP"
  fi
done

# add overall average across datasets
AVG_ROW=$(awk -F',' '
NR==1 {
  ncols = NF
  next
}
$1 != "AVERAGE" {
  for (i=3; i<=NF; i++) {
    if ($i != "" && $i != "nan" && $i ~ /^-?[0-9]+([.][0-9]+)?$/) {
      sum[i] += $i
      count[i]++
    }
  }
}
END {
  printf "AVERAGE,AVERAGE"
  for (i=3; i<=ncols; i++) {
    if (count[i] > 0)
      printf ",%.6f", sum[i] / count[i]
    else
      printf ","
  }
  printf "\n"
}' "$SUMMARY_CSV")

echo "$AVG_ROW" >> "$SUMMARY_CSV"

echo "Saved dataset-average summary to: $SUMMARY_CSV"




