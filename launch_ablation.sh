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
#   ./launch_ablation.sh --project-name my-project --run
#
# NEW (schedule):
#   --lmax-schedule "none decay drop"
#   --lmax-drop-mults "0.5 0.8"
# =============================================================================

set -e

DRY_RUN=true
PRESET="fullgd"
CUSTOM=false
USE_PROTOTYPES=true

MODELS=""
OPTIMIZERS=""
LRS=""
BATCHES=""
LMAX_DECAYS=""
LMAX_SCHEDULES=""
LMAX_DROP_MULTS=""
NUM_DATA=""
STEPS=""
DATASET=""
LOSS=""
CLASSES=""
INIT_SCALE=""
PROJECT_NAME=""

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
    --lmax-decay)
      LMAX_DECAYS="$2"
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
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--run] [--preset fullgd|sgd|adam|momentum|all]"
      echo "       $0 --custom --models \"...\" --optimizers \"...\" --lrs \"...\" [--batches \"...\"] [--lmax-decay \"...\"]"
      exit 1
      ;;
  esac
done

# Prototype registry (model -> seed run ID)
declare -A PROTOTYPES
PROTOTYPES[mlp]="20260108_1246_57_lr0.01000_b8"
PROTOTYPES[cnn]="20260115_1632_56_lr0.01000_b64"
PROTOTYPES[resnet]="20260115_1547_45_lr0.01000_b128"

submit_job() {
  local MODEL=$1
  local OPTIMIZER=$2
  local LR=$3
  local BATCH=$4
  local LMAX_DECAY=${5:-0}
  local EXTRA_EXPORTS=${6:-}
  
  local PROTO=""
  if $USE_PROTOTYPES; then
    PROTO="${PROTOTYPES[$MODEL]}"
  fi
  local JOB_NAME="${MODEL}-${OPTIMIZER}-lr${LR}"
  
  if [[ "$LMAX_DECAY" == "1" ]]; then
    JOB_NAME="${JOB_NAME}-decay"
  fi
  
  local EXPORT_VARS="MODEL=${MODEL},OPTIMIZER=${OPTIMIZER},LR=${LR},BATCH=${BATCH},LMAX_DECAY=${LMAX_DECAY}"
  if [[ -n "$EXTRA_EXPORTS" ]]; then
    EXPORT_VARS="${EXPORT_VARS},${EXTRA_EXPORTS}"
  fi
  if [[ -n "$PROJECT_NAME" ]]; then
    EXPORT_VARS="${EXPORT_VARS},PROJECT_NAME=${PROJECT_NAME}"
  fi
  if [[ -n "$INIT_SCALE" ]]; then
    EXPORT_VARS="${EXPORT_VARS},INIT_SCALE=${INIT_SCALE}"
  fi
  if [[ -n "$PROTO" ]]; then
    EXPORT_VARS="${EXPORT_VARS},TRACK_FEATURE_PROTOTYPES_FROM=${PROTO}"
  fi
  
  if $DRY_RUN; then
    echo "[DRY RUN] sbatch --job-name=${JOB_NAME} --export=${EXPORT_VARS} train_eoss.slurm"
  else
    echo "Submitting: ${JOB_NAME}"
    sbatch --job-name="${JOB_NAME}" --export="${EXPORT_VARS}" train_eoss.slurm
  fi
}

# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

run_fullgd_preset() {
  echo "=== Full-GD Preset: models × LRs × decay ==="
  for MODEL in mlp cnn resnet; do
    for LR in 0.001 0.005 0.01; do
      for LMAX_DECAY in 0 1; do
        submit_job "$MODEL" "fullgd" "$LR" "10000" "$LMAX_DECAY"
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
  local LMAX_DECAYS_LIST="${LMAX_DECAYS:-0}"
  local SCHEDULES_LIST=""
  local DROP_MULTS_LIST="${LMAX_DROP_MULTS:-0.5}"
  local NUM_DATA_VAL="${NUM_DATA:-10000}"

  if [[ -n "$LMAX_SCHEDULES" ]]; then
    SCHEDULES_LIST="$LMAX_SCHEDULES"
  elif [[ -n "$LMAX_DECAYS" ]]; then
    SCHEDULES_LIST=""
    for d in $LMAX_DECAYS; do
      if [[ "$d" == "1" ]]; then
        SCHEDULES_LIST="${SCHEDULES_LIST} drop"
      else
        SCHEDULES_LIST="${SCHEDULES_LIST} none"
      fi
    done
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
            local DECAY_VALUE="0"
            if [[ "$SCHEDULE" == "decay" || "$SCHEDULE" == "drop" ]]; then
              DECAY_VALUE="1"
            fi
            local DROP_MULTS_FOR_SCHEDULE="1"
            if [[ "$SCHEDULE" == "drop" ]]; then
              DROP_MULTS_FOR_SCHEDULE="$DROP_MULTS_LIST"
            fi
            for DROP_MULT in $DROP_MULTS_FOR_SCHEDULE; do
              local EXTRA_EXPORTS="NUM_DATA=${NUM_DATA_VAL},LMAX_SCHEDULE=${SCHEDULE}"
              if [[ -n "$STEPS" ]]; then
                EXTRA_EXPORTS="${EXTRA_EXPORTS},STEPS=${STEPS}"
              fi
              if [[ -n "$DATASET" ]]; then
                EXTRA_EXPORTS="${EXTRA_EXPORTS},DATASET=${DATASET}"
              fi
              if [[ -n "$LOSS" ]]; then
                EXTRA_EXPORTS="${EXTRA_EXPORTS},LOSS=${LOSS}"
              fi
              if [[ -n "$CLASSES" ]]; then
                EXTRA_EXPORTS="${EXTRA_EXPORTS},CLASSES=${CLASSES}"
              fi
              if [[ "$SCHEDULE" == "drop" ]]; then
                EXTRA_EXPORTS="${EXTRA_EXPORTS},LMAX_DROP_MULT=${DROP_MULT}"
              fi
              submit_job "$MODEL" "$OPTIMIZER" "$LR" "$BATCH" "$DECAY_VALUE" "$EXTRA_EXPORTS"
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
