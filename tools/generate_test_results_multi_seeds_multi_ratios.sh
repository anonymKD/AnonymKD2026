#!/bin/bash
set -e
set -x

# ------------------------
# Parse arguments
# ------------------------
KD="$1"
MODEL="$2"
EXP_SUFFIX="$3"
shift 3
PY_ARGS=("$@")

if [ "$KD" == "base" ]; then
  KD_LOSS_WEIGHTS=(1)
else
  KD_LOSS_WEIGHTS=(0.1 1 10)
fi


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

# final global summary across all datasets and all ratios
SUMMARY_CSV="${RESULT_DIR}/${KD}_summary_avg.csv"
> "$SUMMARY_CSV"
FIRST=1

for DATASET in "${DATASETS[@]}"; do
  for KD_WEIGHT in "${KD_LOSS_WEIGHTS[@]}"; do
    # ratio-specific temporary csv containing 3 seed test rows
    DATASET_PREFIX="${KD}_${DATASET}_${KD_WEIGHT}"
    SUMMARY_CSV_TEMP="${RESULT_DIR}/${DATASET_PREFIX}_summary.csv"

    # start fresh for this dataset-ratio
    rm -f "$SUMMARY_CSV_TEMP"

    for i in $(seq 1 "$NUM_RUNS"); do
      sh tools/dist_run.sh \
        "$ENTRY" \
        "$CONFIG" \
        "$MODEL" \
        --experiment_root "$EXP_ROOT" \
        --experiment "$TEST_RESULT_FOLDER" \
        --dataset "$DATASET" \
        --resume "${EXP_ROOT}/${KD}${EXP_SUFFIX}/${DATASET}/${KD_WEIGHT}/seed${i}/last.pth.tar" \
        --prefix_filename "$DATASET_PREFIX" \
        "${PY_ARGS[@]}"
    done

    # add average row to this dataset-ratio temp csv
    python tools/add_avg_row.py --csv "$SUMMARY_CSV_TEMP"

    # initialize final summary header once
    if [ "$FIRST" -eq 1 ]; then
      HEADER=$(head -n 1 "$SUMMARY_CSV_TEMP")
      echo "dataset,kd_loss_weight,${HEADER}" > "$SUMMARY_CSV"
      FIRST=0
    fi

    # append only the AVG row from this dataset-ratio temp csv
    AVG_LINE=$(awk -F',' '$1=="AVG" {print; exit}' "$SUMMARY_CSV_TEMP")
    if [ -n "$AVG_LINE" ]; then
      echo "${DATASET},${KD_WEIGHT},${AVG_LINE}" >> "$SUMMARY_CSV"
    else
      echo "Warning: no AVG row found in $SUMMARY_CSV_TEMP"
    fi
  done
done

# add overall average across all dataset-ratio rows
AVG_ROW=$(awk -F',' '
NR==1 {
  ncols = NF
  next
}
$1 != "AVERAGE" {
  for (i=4; i<=NF; i++) {
    if ($i != "" && $i != "nan" && $i ~ /^-?[0-9]+([.][0-9]+)?$/) {
      sum[i] += $i
      count[i]++
    }
  }
}
END {
  printf "AVERAGE,AVERAGE,AVERAGE"
  for (i=4; i<=ncols; i++) {
    if (count[i] > 0)
      printf ",%.6f", sum[i] / count[i]
    else
      printf ","
  }
  printf "\n"
}' "$SUMMARY_CSV")

echo "$AVG_ROW" >> "$SUMMARY_CSV"

echo "Saved dataset-ratio average summary to: $SUMMARY_CSV"

# -------------------------------------------------------
# Create max-summary CSV: best ratio per dataset by AUC-PRC
# -------------------------------------------------------
MAX_SUMMARY_CSV="${RESULT_DIR}/${KD}_max_summary.csv"
> "$MAX_SUMMARY_CSV"

# detect auc-prc column from header
AUC_COL=$(awk -F',' '
NR==1 {
  for (i=1; i<=NF; i++) {
    name=$i
    gsub(/^[ \t]+|[ \t]+$/, "", name)
    low=tolower(name)
    if (low=="auc_prc" || low=="auc-prc" || low=="auprc" || low=="aucpr" || low=="test_auc_prc" || low=="val_auc_prc") {
      print i
      exit
    }
  }
}
' "$SUMMARY_CSV")

if [ -z "$AUC_COL" ]; then
  echo "Error: could not find AUC-PRC column in $SUMMARY_CSV"
  exit 1
fi

# copy header
head -n 1 "$SUMMARY_CSV" > "$MAX_SUMMARY_CSV"

# for each dataset, keep row with maximum AUC-PRC
for DATASET in "${DATASETS[@]}"; do
  BEST_ROW=$(awk -F',' -v ds="$DATASET" -v auc_col="$AUC_COL" '
  $1 == ds {
    val=$auc_col
    gsub(/^[ \t]+|[ \t]+$/, "", val)
    if (val != "" && val != "nan" && val ~ /^-?[0-9]+([.][0-9]+)?$/) {
      if (!found || val+0 > best+0) {
        best = val+0
        best_row = $0
        found = 1
      }
    }
  }
  END {
    if (found) print best_row
  }
  ' "$SUMMARY_CSV")

  if [ -n "$BEST_ROW" ]; then
    echo "$BEST_ROW" >> "$MAX_SUMMARY_CSV"
  else
    echo "Warning: no valid AUC-PRC row found for dataset $DATASET"
  fi
done

# add overall average across best-per-dataset rows
MAX_AVG_ROW=$(awk -F',' '
NR==1 {
  ncols = NF
  next
}
$1 != "AVERAGE" {
  for (i=4; i<=NF; i++) {
    if ($i != "" && $i != "nan" && $i ~ /^-?[0-9]+([.][0-9]+)?$/) {
      sum[i] += $i
      count[i]++
    }
  }
}
END {
  printf "AVERAGE,BEST_RATIO,AVERAGE"
  for (i=4; i<=ncols; i++) {
    if (count[i] > 0)
      printf ",%.6f", sum[i] / count[i]
    else
      printf ","
  }
  printf "\n"
}
' "$MAX_SUMMARY_CSV")

echo "$MAX_AVG_ROW" >> "$MAX_SUMMARY_CSV"

echo "Saved max-summary by best AUC-PRC to: $MAX_SUMMARY_CSV"

