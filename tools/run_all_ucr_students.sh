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
TEACHER_ARCHITECTURE="ts_lstm_100_3"

CONFIG="configs/train_ts_student.yaml"
EXP_ROOT_BASE="exp_ts_students"
EXP_ROOT_BASE_TEACHER="exp_ts_teachers"

EXP_ROOT="${EXP_ROOT_BASE}/${MODEL}/${KD}"
TEACHER_CKPT_BASE="${EXP_ROOT_BASE_TEACHER}/${TEACHER_ARCHITECTURE}"

if [ "$KD" == "base" ]; then
  KD_METHOD=""
else
  KD_METHOD="$KD"
fi

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
    --teacher-model "$TEACHER_ARCHITECTURE" \
    --kd  "$KD_METHOD" \
    --generative_prior_kwargs "generative_prior_ckpt=${TEACHER_CKPT_BASE}_LD/${DATASET}/best.pth.tar"
    
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

# ------------------------
# Compute column averages
# ------------------------

AVG_ROW=$(awk -F',' '
NR==1 {
  for (i=2; i<=NF; i++) {
    header[i]=$i
  }
  next
}
{
  for (i=2; i<=NF; i++) {
    if ($i != "" && $i != "nan") {
      sum[i] += $i
      count[i]++
    }
  }
}
END {
  printf "AVERAGE"
  for (i=2; i<=length(header)+1; i++) {
    if (count[i] > 0)
      printf ",%.6f", sum[i]/count[i]
    else
      printf ","
  }
  printf "\n"
}
' "$MERGED_CSV")

echo "$AVG_ROW" >> "$MERGED_CSV"

echo "Merged summary with averages saved to: $MERGED_CSV"

