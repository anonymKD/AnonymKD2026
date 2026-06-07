#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
KD="$1"
MODEL="$2"
shift 2
PY_ARGS=("$@")     # Capture remaining args safely


# MODEL="ts_lstm_100_3"
TEACHER_ARCHITECTURE="ts_lstm_100_3"

EXP_ROOT_BASE="exp_ts_students_ms"
EXP_ROOT_BASE_TEACHER="exp_ts_teachers"

CONFIG="configs/train_ts_student.yaml"

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
  sh tools/bulk_run.sh \
    "$CONFIG" \
    "$MODEL" \
    "$EXP_ROOT" \
    "$DATASET" \
    --dataset "$DATASET" \
    --teacher-ckpt "${TEACHER_CKPT_BASE}/${DATASET}/best.pth.tar" \
    --teacher-model "$TEACHER_ARCHITECTURE" \
    --kd  "$KD_METHOD" \
    --generative_prior_kwargs "generative_prior_ckpt=${TEACHER_CKPT_BASE}_LD/${DATASET}/best.pth.tar" \
    # --val_loss_monitor_metric auc_prc  \
    "${PY_ARGS[@]}" 
    
done

# ------------------------
# Merge only the AVERAGE row from each dataset-level all_seeds_summary.csv
# ------------------------
MERGED_CSV="${EXP_ROOT}/all_val_result_summary.csv"
FIRST=1

> "$MERGED_CSV"

for DATASET in "${DATASETS[@]}"; do
  FILE="${EXP_ROOT}/${DATASET}/all_seeds_summary.csv"

  if [ -f "$FILE" ]; then
    if [ "$FIRST" -eq 1 ]; then
      HEADER=$(head -n 1 "$FILE")
      echo "dataset,${HEADER}" > "$MERGED_CSV"
      FIRST=0
    fi

    AVG_LINE=$(awk -F',' '$1=="AVG" {print; exit}' "$FILE")

    if [ -n "$AVG_LINE" ]; then
      echo "${DATASET},${AVG_LINE}" >> "$MERGED_CSV"
    else
      echo "Warning: no AVERAGE row found in $FILE"
    fi
  else
    echo "Warning: missing $FILE"
  fi
done

# ------------------------
# Compute overall average across datasets
# ------------------------
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
      printf ",%.6f", sum[i]/count[i]
    else
      printf ","
  }
  printf "\n"
}
' "$MERGED_CSV")

echo "$AVG_ROW" >> "$MERGED_CSV"

echo "Merged summary with averages saved to: $MERGED_CSV"

