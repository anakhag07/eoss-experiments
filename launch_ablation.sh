#!/bin/bash
# =============================================================================
# ABLATION EXPERIMENT LAUNCHER
# =============================================================================
# Usage:
#   ./launch_ablation.sh                    # Dry run (shows what would be submitted)
#   ./launch_ablation.sh --run              # Actually submit jobs
#   ./launch_ablation.sh --preset full_gd   # Run full-GD preset
#   ./launch_ablation.sh --preset sgd       # Run SGD preset
#   ./launch_ablation.sh --preset all       # Run all presets
# =============================================================================

set -e

DRY_RUN=true
PRESET="full_gd"

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
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--run] [--preset full_gd|sgd|adam|all]"
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
  
  local PROTO="${PROTOTYPES[$MODEL]}"
  local JOB_NAME="${MODEL}-${OPTIMIZER}-lr${LR}"
  
  if [[ "$LMAX_DECAY" == "1" ]]; then
    JOB_NAME="${JOB_NAME}-decay"
  fi
  
  local EXPORT_VARS="MODEL=${MODEL},OPTIMIZER=${OPTIMIZER},LR=${LR},BATCH=${BATCH},LMAX_DECAY=${LMAX_DECAY}"
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

run_full_gd_preset() {
  echo "=== Full-GD Preset: models × LRs × decay ==="
  for MODEL in mlp cnn resnet; do
    for LR in 0.001 0.005 0.01; do
      for LMAX_DECAY in 0 1; do
        submit_job "$MODEL" "full_gd" "$LR" "10000" "$LMAX_DECAY"
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
# MAIN
# =============================================================================

echo "=============================================="
echo "DRY_RUN: $DRY_RUN (use --run to submit)"
echo "PRESET: $PRESET"
echo "=============================================="
echo ""

case "$PRESET" in
  full_gd)
    run_full_gd_preset
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
  all)
    run_full_gd_preset
    echo ""
    run_sgd_preset
    echo ""
    run_adam_preset
    echo ""
    run_momentum_preset
    ;;
  *)
    echo "Unknown preset: $PRESET"
    echo "Available presets: full_gd, sgd, adam, momentum, all"
    exit 1
    ;;
esac

echo ""
if $DRY_RUN; then
  echo "This was a dry run. Use --run to actually submit jobs."
fi
