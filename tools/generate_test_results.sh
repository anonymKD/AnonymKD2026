#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
KD="$1"
shift 1
PY_ARGS=("$@")     # Capture remaining args safely


MODEL="ts_lstm_32_2"

ENTRY="tools/test.py"
CONFIG="configs/test_ts_student.yaml"
EXP_ROOT_BASE="exp_ts_students"
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

for DATASET in "${DATASETS[@]}"; do
  sh tools/dist_run.sh \
    "$ENTRY" \
    "$CONFIG" \
    "$MODEL" \
    --experiment_root "$EXP_ROOT" \
    --experiment "$TEST_RESULT_FOLDER" \
    --dataset "$DATASET" \
    --resume "${EXP_ROOT}/${KD}/${DATASET}/best.pth.tar" \
    --prefix_filename "${KD}"
    
done

SUMMARY_CSV="${EXP_ROOT_BASE}/${MODEL}/${TEST_RESULT_FOLDER}/${KD}_summary.csv"

if [ -f "$SUMMARY_CSV" ]; then
  AVG_ROW=$(awk -F',' '
  NR==1 {
    ncols = NF
    next
  }
  $1 != "AVERAGE" {
    for (i=2; i<=NF; i++) {
      if ($i != "" && $i != "nan" && $i ~ /^-?[0-9]+([.][0-9]+)?$/) {
        sum[i] += $i
        count[i]++
      }
    }
  }
  END {
    printf "AVERAGE"
    for (i=2; i<=ncols; i++) {
      if (count[i] > 0)
        printf ",%.6f", sum[i]/count[i]
      else
        printf ","
    }
    printf "\n"
  }' "$SUMMARY_CSV")

  TMP_FILE="${SUMMARY_CSV}.tmp"
  awk -F',' '$1 != "AVERAGE"' "$SUMMARY_CSV" > "$TMP_FILE"
  mv "$TMP_FILE" "$SUMMARY_CSV"

  echo "$AVG_ROW" >> "$SUMMARY_CSV"
  echo "Average row appended to: $SUMMARY_CSV"
else
  echo "Warning: summary file not found: $SUMMARY_CSV"
fi