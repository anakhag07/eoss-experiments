#!/bin/bash
# =============================================================================
# ABLATION EXPERIMENT LAUNCHER
# =============================================================================
# Usage:
#   ./launch_ablation.sh                          # Dry run (shows what would be submitted)
#   ./launch_ablation.sh --run                    # Actually submit jobs
#   ./launch_ablation.sh --preset fullgd          # Run full-GD preset
#   ./launch_ablation.sh --preset sgd             # Run SGD preset
#   ./launch_ablation.sh --preset all             # Run all presets
#   ./launch_ablation.sh --custom --models "mlp cnn" --optimizers "sgd adam" \
#     --lrs "0.001 0.005" --batches "8 32" --run   # Custom grid
#   ./launch_ablation.sh --custom --optional-flags "--cpu --disable-wandb"
#   ./launch_ablation.sh --project-name my-project --run
#   ./launch_ablation.sh --preset sgd --config-path configs/your_base.json --run
#
# NEW (schedule):
#   --lmax-schedule "none drop"
#   --lmax-drop-mults "0.5 0.8"
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_FILE="${SCRIPT_DIR}/train_eoss.slurm"

DRY_RUN=true
PRESET="fullgd"
CUSTOM=false
USE_PROTOTYPES=false

MODELS=""
OPTIMIZERS=""
LRS=""
BATCHES=""
LMAX_SCHEDULES=""
LMAX_DROP_MULTS=""
NUM_DATA=""
STEPS=""
DATASET=""
LOSS=""
CLASSES=""
INIT_SCALE=""
PROJECT_NAME=""
CONFIG_PATH=""
OPTIONAL_FLAGS=""
INPUT_PROTOTYPES_MODES=""
INPUT_PROTOTYPE_SOURCES=""
INPUT_BOUNDARY_COUNTS=""
INPUT_INLIERS_COUNTS=""
INPUT_X_OUTLIER_COUNTS=""
INPUT_Y_OUTLIER_COUNTS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --run)
      DRY_RUN=false
      shift
      ;;
    --preset)
      PRESET="$2"
      shift 2
      ;;
    --custom)
      CUSTOM=true
      PRESET="custom"
      shift
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --optimizers)
      OPTIMIZERS="$2"
      shift 2
      ;;
    --lrs)
      LRS="$2"
      shift 2
      ;;
    --batches)
      BATCHES="$2"
      shift 2
      ;;
    --lmax-schedule)
      LMAX_SCHEDULES="$2"
      shift 2
      ;;
    --lmax-drop-mults)
      LMAX_DROP_MULTS="$2"
      shift 2
      ;;
    --num-data)
      NUM_DATA="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --loss)
      LOSS="$2"
      shift 2
      ;;
    --classes)
      CLASSES="$2"
      shift 2
      ;;
    --init-scale)
      INIT_SCALE="$2"
      shift 2
      ;;
    --no-prototypes)
      USE_PROTOTYPES=false
      shift
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --optional-flags)
      OPTIONAL_FLAGS="$2"
      shift 2
      ;;
    --input-prototypes-modes)
      INPUT_PROTOTYPES_MODES="$2"
      shift 2
      ;;
    --input-prototype-sources)
      INPUT_PROTOTYPE_SOURCES="$2"
      shift 2
      ;;
    --input-boundary-counts)
      INPUT_BOUNDARY_COUNTS="$2"
      shift 2
      ;;
    --input-inliers-counts)
      INPUT_INLIERS_COUNTS="$2"
      shift 2
      ;;
    --input-x-outlier-counts)
      INPUT_X_OUTLIER_COUNTS="$2"
      shift 2
      ;;
    --input-y-outlier-counts)
      INPUT_Y_OUTLIER_COUNTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--run] [--preset fullgd|sgd|adam|momentum|all]"
      echo "       $0 --custom --models \"...\" --optimizers \"...\" --lrs \"...\" [--batches \"...\"] [--lmax-schedule \"...\"]"
      exit 1
      ;;
  esac
done

submit_job() {
  local MODEL=$1
  local OPTIMIZER=$2
  local LR=$3
  local BATCH=$4
  local LMAX_SCHEDULE=${5:-none}
  local EXTRA_EXPORTS=${6:-}
  local JOB_NAME="${MODEL}-${OPTIMIZER}-lr${LR}"
  
  if [[ "$LMAX_SCHEDULE" == "drop" ]]; then
    JOB_NAME="${JOB_NAME}-drop"
  fi

  local EXPORT_VARS="MODEL=${MODEL},OPTIMIZER=${OPTIMIZER},LR=${LR},BATCH=${BATCH},LMAX_SCHEDULE=${LMAX_SCHEDULE}"
  if [[ -n "$EXTRA_EXPORTS" ]]; then
    EXPORT_VARS="${EXPORT_VARS},${EXTRA_EXPORTS}"
  fi
  if [[ -n "$PROJECT_NAME" ]]; then
    EXPORT_VARS="${EXPORT_VARS},PROJECT_NAME=${PROJECT_NAME}"
  fi
  if [[ -n "$CONFIG_PATH" ]]; then
    EXPORT_VARS="${EXPORT_VARS},CONFIG_PATH=${CONFIG_PATH}"
  fi
  if [[ -n "$INIT_SCALE" ]]; then
    EXPORT_VARS="${EXPORT_VARS},INIT_SCALE=${INIT_SCALE}"
  fi
  
  if $DRY_RUN; then
    echo "[DRY RUN] sbatch --job-name=${JOB_NAME} --export=${EXPORT_VARS} ${SLURM_FILE}"
  else
    echo "Submitting: ${JOB_NAME}"
    sbatch --job-name="${JOB_NAME}" --export="${EXPORT_VARS}" "${SLURM_FILE}"
  fi
}

# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

run_fullgd_preset() {
  echo "=== Full-GD Preset: models × LRs × schedule ==="
  for MODEL in mlp cnn resnet; do
    for LR in 0.001 0.005 0.01; do
      for LMAX_SCHEDULE in none drop; do
        submit_job "$MODEL" "fullgd" "$LR" "10000" "$LMAX_SCHEDULE"
      done
    done
  done
}

run_sgd_preset() {
  echo "=== SGD Preset: LRs × batch sizes ==="
  for LR in 0.001 0.005 0.01 0.05; do
    for BATCH in 8 32 128; do
      submit_job "mlp" "sgd" "$LR" "$BATCH"
    done
  done
}

run_adam_preset() {
  echo "=== Adam Preset: LRs × batch sizes ==="
  for LR in 0.001 0.005 0.01; do
    for BATCH in 8 32 64; do
      submit_job "mlp" "adam" "$LR" "$BATCH"
    done
  done
}

run_momentum_preset() {
  echo "=== Momentum Preset: LRs × batch sizes ==="
  for LR in 0.001 0.005 0.01; do
    for BATCH in 8 32 64; do
      submit_job "mlp" "momentum" "$LR" "$BATCH"
    done
  done
}

# =============================================================================
# CUSTOM GRID
# =============================================================================

run_custom_grid() {
  echo "=== Custom Grid ==="

  local MODELS_LIST="${MODELS:-mlp}"
  local OPTIMIZERS_LIST="${OPTIMIZERS:-sgd}"
  local LRS_LIST="${LRS:-0.01}"
  local SCHEDULES_LIST=""
  local DROP_MULTS_LIST="${LMAX_DROP_MULTS:-0.5}"
  local NUM_DATA_VAL="${NUM_DATA:-10000}"
  local LOSS_LIST="${LOSS:-ce}"
  local INPUT_PROTOTYPES_MODES_LIST="${INPUT_PROTOTYPES_MODES:-train}"
  local INPUT_PROTOTYPE_SOURCES_LIST="$INPUT_PROTOTYPE_SOURCES"
  local INPUT_BOUNDARY_COUNTS_LIST="${INPUT_BOUNDARY_COUNTS:-__UNSET__}"
  local INPUT_INLIERS_COUNTS_LIST="${INPUT_INLIERS_COUNTS:-__UNSET__}"
  local INPUT_X_OUTLIER_COUNTS_LIST="${INPUT_X_OUTLIER_COUNTS:-__UNSET__}"
  local INPUT_Y_OUTLIER_COUNTS_LIST="${INPUT_Y_OUTLIER_COUNTS:-__UNSET__}"
  local HAS_PROTO_COUNT_GRID=0

  if [[ -n "$INPUT_BOUNDARY_COUNTS" || -n "$INPUT_INLIERS_COUNTS" || -n "$INPUT_X_OUTLIER_COUNTS" || -n "$INPUT_Y_OUTLIER_COUNTS" ]]; then
    HAS_PROTO_COUNT_GRID=1
  fi
  if [[ -z "$INPUT_PROTOTYPE_SOURCES_LIST" ]]; then
    if [[ "$HAS_PROTO_COUNT_GRID" == "1" ]]; then
      INPUT_PROTOTYPE_SOURCES_LIST="generate"
    else
      INPUT_PROTOTYPE_SOURCES_LIST="none"
    fi
  fi
  if [[ "$HAS_PROTO_COUNT_GRID" == "0" ]]; then
    for INPUT_PROTOTYPE_SOURCE_VAL in $INPUT_PROTOTYPE_SOURCES_LIST; do
      if [[ "$INPUT_PROTOTYPE_SOURCE_VAL" != "none" ]]; then
        echo "Input prototype sources '${INPUT_PROTOTYPE_SOURCES_LIST}' require at least one of --input-boundary-counts, --input-inliers-counts, --input-x-outlier-counts, or --input-y-outlier-counts."
        exit 1
      fi
    done
  fi

  if [[ -n "$LMAX_SCHEDULES" ]]; then
    SCHEDULES_LIST="$LMAX_SCHEDULES"
  else
    SCHEDULES_LIST="none"
  fi

  for MODEL in $MODELS_LIST; do
    for OPTIMIZER in $OPTIMIZERS_LIST; do
      local BATCHES_LIST=""
      if [[ -n "$BATCHES" ]]; then
        BATCHES_LIST="$BATCHES"
      elif [[ "$OPTIMIZER" == "fullgd" ]]; then
        BATCHES_LIST="$NUM_DATA_VAL"
      else
        BATCHES_LIST="128"
      fi

      for LR in $LRS_LIST; do
        for BATCH in $BATCHES_LIST; do
          for SCHEDULE in $SCHEDULES_LIST; do
            if [[ "$SCHEDULE" != "none" && "$SCHEDULE" != "drop" ]]; then
              echo "Unsupported --lmax-schedule value: $SCHEDULE"
              exit 1
            fi
            local DROP_MULTS_FOR_SCHEDULE="1"
            if [[ "$SCHEDULE" == "drop" ]]; then
              DROP_MULTS_FOR_SCHEDULE="$DROP_MULTS_LIST"
            fi
            for DROP_MULT in $DROP_MULTS_FOR_SCHEDULE; do
              for LOSS_VAL in $LOSS_LIST; do
                local BASE_EXPORTS="NUM_DATA=${NUM_DATA_VAL},LMAX_SCHEDULE=${SCHEDULE}"
                if [[ -n "$STEPS" ]]; then
                  BASE_EXPORTS="${BASE_EXPORTS},STEPS=${STEPS}"
                fi
                if [[ -n "$DATASET" ]]; then
                  BASE_EXPORTS="${BASE_EXPORTS},DATASET=${DATASET}"
                fi
                if [[ -n "$LOSS_VAL" ]]; then
                  BASE_EXPORTS="${BASE_EXPORTS},LOSS=${LOSS_VAL}"
                fi
                if [[ -n "$CLASSES" ]]; then
                  BASE_EXPORTS="${BASE_EXPORTS},CLASSES=${CLASSES}"
                fi
                if [[ -n "$OPTIONAL_FLAGS" ]]; then
                  BASE_EXPORTS="${BASE_EXPORTS},OPTIONAL_FLAGS=${OPTIONAL_FLAGS}"
                fi
                if [[ "$SCHEDULE" == "drop" ]]; then
                  BASE_EXPORTS="${BASE_EXPORTS},LMAX_DROP_MULT=${DROP_MULT}"
                fi

                for INPUT_PROTOTYPES_MODE_VAL in $INPUT_PROTOTYPES_MODES_LIST; do
                  for INPUT_PROTOTYPE_SOURCE_VAL in $INPUT_PROTOTYPE_SOURCES_LIST; do
                    local PROTO_EXPORTS="${BASE_EXPORTS},INPUT_PROTOTYPES_MODE=${INPUT_PROTOTYPES_MODE_VAL},INPUT_PROTOTYPE_SOURCE=${INPUT_PROTOTYPE_SOURCE_VAL}"
                    if [[ "$INPUT_PROTOTYPE_SOURCE_VAL" == "none" ]]; then
                      submit_job "$MODEL" "$OPTIMIZER" "$LR" "$BATCH" "$SCHEDULE" "$PROTO_EXPORTS"
                      continue
                    fi

                    for INPUT_BOUNDARY_COUNT_VAL in $INPUT_BOUNDARY_COUNTS_LIST; do
                      for INPUT_INLIERS_COUNT_VAL in $INPUT_INLIERS_COUNTS_LIST; do
                        for INPUT_X_OUTLIER_COUNT_VAL in $INPUT_X_OUTLIER_COUNTS_LIST; do
                          for INPUT_Y_OUTLIER_COUNT_VAL in $INPUT_Y_OUTLIER_COUNTS_LIST; do
                            local COUNT_EXPORTS="$PROTO_EXPORTS"
                            local HAS_COUNT=0
                            if [[ "$INPUT_BOUNDARY_COUNT_VAL" != "__UNSET__" ]]; then
                              COUNT_EXPORTS="${COUNT_EXPORTS},INPUT_BOUNDARY=${INPUT_BOUNDARY_COUNT_VAL}"
                              HAS_COUNT=1
                            fi
                            if [[ "$INPUT_INLIERS_COUNT_VAL" != "__UNSET__" ]]; then
                              COUNT_EXPORTS="${COUNT_EXPORTS},INPUT_INLIERS=${INPUT_INLIERS_COUNT_VAL}"
                              HAS_COUNT=1
                            fi
                            if [[ "$INPUT_X_OUTLIER_COUNT_VAL" != "__UNSET__" ]]; then
                              COUNT_EXPORTS="${COUNT_EXPORTS},INPUT_X_OUTLIERS=${INPUT_X_OUTLIER_COUNT_VAL}"
                              HAS_COUNT=1
                            fi
                            if [[ "$INPUT_Y_OUTLIER_COUNT_VAL" != "__UNSET__" ]]; then
                              COUNT_EXPORTS="${COUNT_EXPORTS},INPUT_Y_OUTLIERS=${INPUT_Y_OUTLIER_COUNT_VAL}"
                              HAS_COUNT=1
                            fi
                            if [[ "$HAS_COUNT" == "1" ]]; then
                              submit_job "$MODEL" "$OPTIMIZER" "$LR" "$BATCH" "$SCHEDULE" "$COUNT_EXPORTS"
                            fi
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

# =============================================================================
# MAIN
# =============================================================================

echo "=============================================="
echo "DRY_RUN: $DRY_RUN (use --run to submit)"
echo "PRESET: $PRESET"
echo "=============================================="
echo ""

case "$PRESET" in
  fullgd)
    run_fullgd_preset
    ;;
  sgd)
    run_sgd_preset
    ;;
  adam)
    run_adam_preset
    ;;
  momentum)
    run_momentum_preset
    ;;
  custom)
    run_custom_grid
    ;;
  all)
    run_fullgd_preset
    echo ""
    run_sgd_preset
    echo ""
    run_adam_preset
    echo ""
    run_momentum_preset
    ;;
  *)
    echo "Unknown preset: $PRESET"
    echo "Available presets: fullgd, sgd, adam, momentum, all"
    exit 1
    ;;
esac

echo ""
if $DRY_RUN; then
  echo "This was a dry run. Use --run to actually submit jobs."
fi
