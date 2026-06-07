#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
TEACHER_ARCHITECTURE="$1"
MODEL="$2"
KD="$3"
EXP_SUFFIX="$4"
shift 4
PY_ARGS=("$@")     # Capture remaining args safely

if [ "$KD" == "base" ]; then
  KD_LOSS_WEIGHTS=(1)
else
  KD_LOSS_WEIGHTS=(0.1 1 10)
fi

# MODEL="ts_lstm_32_2"
# TEACHER_ARCHITECTURE="ts_lstm_100_3"

EXP_ROOT_BASE="exp_ts_students_ms"
EXP_ROOT_BASE_TEACHER="exp_ts_teachers"

CONFIG="configs/train_ts_student.yaml"

BASE_EXP_ROOT="${EXP_ROOT_BASE}/${MODEL}/${KD}${EXP_SUFFIX}"
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

# ------------------------
# Run all datasets for all KD weights
# Desired structure:
#   mse/Computers/0.1/seed1
#   mse/Computers/1/seed1
#   mse/Computers/10/seed1
# ------------------------
for DATASET in "${DATASETS[@]}"; do
  DATASET_EXP_ROOT="${BASE_EXP_ROOT}/${DATASET}"

  for KD_WEIGHT in "${KD_LOSS_WEIGHTS[@]}"; do
    sh tools/bulk_run.sh \
      "$CONFIG" \
      "$MODEL" \
      "$DATASET_EXP_ROOT" \
      "$KD_WEIGHT" \
      --dataset "$DATASET" \
      --teacher-ckpt "${TEACHER_CKPT_BASE}/${DATASET}/best.pth.tar" \
      --teacher-model "$TEACHER_ARCHITECTURE" \
      --kd "$KD_METHOD" \
      --generative_prior_kwargs "generative_prior_ckpt=${TEACHER_CKPT_BASE}_LD/${DATASET}/best.pth.tar" \
      --kd_loss_weight "$KD_WEIGHT" \
      "${PY_ARGS[@]}"
  done
done

# ------------------------
# Merge the AVG row corresponding to the best KD weight
# for each dataset, based on maximum Top-1
# ------------------------
MERGED_CSV="${BASE_EXP_ROOT}/all_val_result_summary.csv"
FIRST=1

> "$MERGED_CSV"

for DATASET in "${DATASETS[@]}"; do
  BEST_LINE=""
  BEST_WEIGHT=""
  BEST_TOP1=""

  for KD_WEIGHT in "${KD_LOSS_WEIGHTS[@]}"; do
    FILE="${BASE_EXP_ROOT}/${DATASET}/${KD_WEIGHT}/all_seeds_summary.csv"

    if [ -f "$FILE" ]; then
      if [ "$FIRST" -eq 1 ]; then
        HEADER=$(head -n 1 "$FILE")
        echo "dataset,kd_loss_weight,${HEADER}" > "$MERGED_CSV"
        FIRST=0
      fi

      # find column index of top1 from header
      TOP1_COL=$(awk -F',' '
        NR==1 {
          for (i=1; i<=NF; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            low=tolower($i)
            if (low=="top1" || low=="top-1" || low=="val_top1" || low=="best_top1" || low=="acc" || low=="accuracy") {
              print i
              exit
            }
          }
        }
      ' "$FILE")

      if [ -z "$TOP1_COL" ]; then
        echo "Warning: could not find Top-1 column in $FILE"
        continue
      fi

      AVG_LINE=$(awk -F',' '$1=="AVG" {print; exit}' "$FILE")

      if [ -n "$AVG_LINE" ]; then
        CUR_TOP1=$(echo "$AVG_LINE" | awk -F',' -v col="$TOP1_COL" '{gsub(/^[ \t]+|[ \t]+$/, "", $col); print $col}')

        if [ -n "$CUR_TOP1" ] && [[ "$CUR_TOP1" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
          if [ -z "$BEST_TOP1" ] || awk "BEGIN {exit !($CUR_TOP1 > $BEST_TOP1)}"; then
            BEST_TOP1="$CUR_TOP1"
            BEST_LINE="$AVG_LINE"
            BEST_WEIGHT="$KD_WEIGHT"
          fi
        else
          echo "Warning: invalid Top-1 value in $FILE -> $CUR_TOP1"
        fi
      else
        echo "Warning: no AVG row found in $FILE"
      fi
    else
      echo "Warning: missing $FILE"
    fi
  done

  if [ -n "$BEST_LINE" ]; then
    echo "${DATASET},${BEST_WEIGHT},${BEST_LINE}" >> "$MERGED_CSV"
  else
    echo "Warning: no valid AVG row found for dataset $DATASET"
  fi
done

# ------------------------
# Compute overall average across selected best rows
# ------------------------
AVG_ROW=$(awk -F',' '
NR==1 {
  ncols = NF
  next
}
$1 != "AVERAGE" && $1 != "dataset" {
  for (i=4; i<=NF; i++) {
    if ($i != "" && $i != "nan" && $i ~ /^-?[0-9]+([.][0-9]+)?$/) {
      sum[i] += $i
      count[i]++
    }
  }
}
END {
  printf "AVERAGE,BEST_WEIGHT,AVERAGE"
  for (i=4; i<=ncols; i++) {
    if (count[i] > 0)
      printf ",%.6f", sum[i]/count[i]
    else
      printf ","
  }
  printf "\n"
}
' "$MERGED_CSV")

echo "$AVG_ROW" >> "$MERGED_CSV"

echo "Merged summary using best kd_loss_weight per dataset saved to: $MERGED_CSV"