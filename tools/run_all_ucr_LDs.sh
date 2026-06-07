#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
TEACHER_ARCHITECTURE="$1"
shift 1
PY_ARGS=("$@")     # Capture remaining args safely

# TEACHER_ARCHITECTURE="ts_lstm_100_3"

CONFIG="configs/train_ts_LD.yaml"
MODEL="latentdiffusion"
EXP_ROOT_BASE="exp_ts_teachers"

EXP_ROOT="${EXP_ROOT_BASE}/${TEACHER_ARCHITECTURE}_LD"  #omit ts_ prefix
TEACHER_CKPT_BASE="${EXP_ROOT_BASE}/${TEACHER_ARCHITECTURE}"

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
  sh tools/dist_train.sh \
    "$CONFIG" \
    "$MODEL" \
    "$EXP_ROOT" \
    "$DATASET" \
    --dataset "$DATASET" \
    --teacher-ckpt "${TEACHER_CKPT_BASE}/${DATASET}/best.pth.tar" \
    --teacher-model "$TEACHER_ARCHITECTURE"
done

MERGED_CSV="${EXP_ROOT}/all_result_summary.csv"
FIRST=1

> "$MERGED_CSV"

for DATASET in "${DATASETS[@]}"; do
  FILE="${EXP_ROOT}/${DATASET}/summary.csv"

  if [ -f "$FILE" ]; then
    if [ "$FIRST" -eq 1 ]; then
      HEADER=$(head -n 1 "$FILE")
      echo "dataset,${HEADER}" > "$MERGED_CSV"
      tail -n +2 "$FILE" | sed "s/^/${DATASET},/" >> "$MERGED_CSV"
      FIRST=0
    else
      tail -n +2 "$FILE" | sed "s/^/${DATASET},/" >> "$MERGED_CSV"
    fi
  else
    echo "Warning: missing $FILE"
  fi
done

echo "Merged summary saved to: $MERGED_CSV"