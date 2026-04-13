#!/bin/bash

set -euo pipefail

DRY_RUN=true

MODEL="mlp"
OPTIMIZER="fullgd"
DATASET="cifar10_2cls"
LOSS="mse"
CLASSES="1 9"
NUM_DATA="10000"
BATCH=""
INIT_SCALE="0.2"
STEPS=""
PROJECT_NAME=""
CONFIG_PATH=""
CHECKPOINT_EVERY=""
WANDB_TAG=""
OPTIONAL_FLAGS=""

INPUT_PROTOTYPES_MODE=""
INPUT_PROTOTYPE_SOURCE=""
INPUT_BOUNDARY=""
INPUT_INLIERS=""
INPUT_X_OUTLIERS=""
INPUT_Y_OUTLIERS=""

BASELINE_RUN_IDS=""
CONT_STEPS=""
BASE_LRS=""
DROP_MULTS="0.5"

usage() {
  cat <<'EOF'
Usage:
  ./launch_fork_ablation.sh \
    --baseline-run-ids "run1 run2" \
    --cont-steps "42000 31000" \
    --base-lrs "0.01 0.005" \
    --drop-mults "0.8 0.5 0.2"

Options:
  --run                       Submit jobs (default is dry-run)
  --model MODEL              Default: mlp
  --optimizer OPT            Default: fullgd
  --dataset DATASET          Default: cifar10_2cls
  --loss LOSS                Default: mse
  --classes "A B"            Default: "1 9"
  --num-data N               Default: 10000
  --batch N|full             Optional; fullgd defaults to full dataset in train_eoss.slurm
  --init-scale X             Default: 0.2
  --steps N                  Optional; otherwise train_eoss.slurm auto-computes from LR
  --project-name NAME        Optional WANDB_PROJECT override
  --config-path PATH         Optional base config
  --checkpoint-every N       Optional checkpoint cadence
  --wandb-tag TAG            Optional tag for descendant runs
  --optional-flags "..."     Extra training.py flags appended verbatim

  --input-prototypes-mode MODE
  --input-prototype-source SRC
  --input-boundary N
  --input-inliers N
  --input-x-outliers N
  --input-y-outliers N

  --baseline-run-ids "..."   Space-separated run IDs
  --cont-steps "..."         Space-separated continuation steps
  --base-lrs "..."           Space-separated baseline learning rates
  --drop-mults "..."         Space-separated multipliers; default: "0.5"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      DRY_RUN=false
      shift
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --optimizer)
      OPTIMIZER="$2"
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
    --num-data)
      NUM_DATA="$2"
      shift 2
      ;;
    --batch)
      BATCH="$2"
      shift 2
      ;;
    --init-scale)
      INIT_SCALE="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --wandb-tag)
      WANDB_TAG="$2"
      shift 2
      ;;
    --optional-flags)
      OPTIONAL_FLAGS="$2"
      shift 2
      ;;
    --input-prototypes-mode)
      INPUT_PROTOTYPES_MODE="$2"
      shift 2
      ;;
    --input-prototype-source)
      INPUT_PROTOTYPE_SOURCE="$2"
      shift 2
      ;;
    --input-boundary)
      INPUT_BOUNDARY="$2"
      shift 2
      ;;
    --input-inliers)
      INPUT_INLIERS="$2"
      shift 2
      ;;
    --input-x-outliers)
      INPUT_X_OUTLIERS="$2"
      shift 2
      ;;
    --input-y-outliers)
      INPUT_Y_OUTLIERS="$2"
      shift 2
      ;;
    --baseline-run-ids)
      BASELINE_RUN_IDS="$2"
      shift 2
      ;;
    --cont-steps)
      CONT_STEPS="$2"
      shift 2
      ;;
    --base-lrs)
      BASE_LRS="$2"
      shift 2
      ;;
    --drop-mults)
      DROP_MULTS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

read -r -a RUN_IDS_ARRAY <<< "$BASELINE_RUN_IDS"
read -r -a CONT_STEPS_ARRAY <<< "$CONT_STEPS"
read -r -a BASE_LRS_ARRAY <<< "$BASE_LRS"
read -r -a DROP_MULTS_ARRAY <<< "$DROP_MULTS"

if [[ ${#RUN_IDS_ARRAY[@]} -eq 0 || ${#CONT_STEPS_ARRAY[@]} -eq 0 || ${#BASE_LRS_ARRAY[@]} -eq 0 ]]; then
  echo "baseline-run-ids, cont-steps, and base-lrs are all required"
  usage
  exit 1
fi

if [[ ${#RUN_IDS_ARRAY[@]} -ne ${#CONT_STEPS_ARRAY[@]} || ${#RUN_IDS_ARRAY[@]} -ne ${#BASE_LRS_ARRAY[@]} ]]; then
  echo "baseline-run-ids, cont-steps, and base-lrs must have the same number of entries"
  exit 1
fi

if [[ ${#DROP_MULTS_ARRAY[@]} -eq 0 ]]; then
  echo "drop-mults must contain at least one multiplier"
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

submit_job() {
  local run_id="$1"
  local cont_step="$2"
  local base_lr="$3"
  local drop_mult="$4"
  local target_lr
  target_lr=$(awk -v lr="$base_lr" -v m="$drop_mult" 'BEGIN { printf "%.10g", lr * m }')

  local export_vars="MODEL=${MODEL},OPTIMIZER=${OPTIMIZER},DATASET=${DATASET},LOSS=${LOSS},LR=${base_lr},NUM_DATA=${NUM_DATA},CLASSES=${CLASSES},CONT_RUN_ID=${run_id},CONT_STEP=${cont_step},LR_DROP_AT_STEP=${cont_step},LR_DROP_TO=${target_lr},LMAX_SCHEDULE=none"
  local job_name="fork-${MODEL}-${OPTIMIZER}-lr${base_lr}-to${target_lr}-step${cont_step}"

  if [[ -n "$BATCH" ]]; then
    export_vars="${export_vars},BATCH=${BATCH}"
  fi
  if [[ -n "$INIT_SCALE" ]]; then
    export_vars="${export_vars},INIT_SCALE=${INIT_SCALE}"
  fi
  if [[ -n "$STEPS" ]]; then
    export_vars="${export_vars},STEPS=${STEPS}"
  fi
  if [[ -n "$PROJECT_NAME" ]]; then
    export_vars="${export_vars},PROJECT_NAME=${PROJECT_NAME}"
  fi
  if [[ -n "$CONFIG_PATH" ]]; then
    export_vars="${export_vars},CONFIG_PATH=${CONFIG_PATH}"
  fi
  if [[ -n "$CHECKPOINT_EVERY" ]]; then
    export_vars="${export_vars},CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  fi
  if [[ -n "$WANDB_TAG" ]]; then
    export_vars="${export_vars},WANDB_TAG=${WANDB_TAG}"
  fi
  if [[ -n "$OPTIONAL_FLAGS" ]]; then
    export_vars="${export_vars},OPTIONAL_FLAGS=${OPTIONAL_FLAGS}"
  fi
  if [[ -n "$INPUT_PROTOTYPES_MODE" ]]; then
    export_vars="${export_vars},INPUT_PROTOTYPES_MODE=${INPUT_PROTOTYPES_MODE}"
  fi
  if [[ -n "$INPUT_PROTOTYPE_SOURCE" ]]; then
    export_vars="${export_vars},INPUT_PROTOTYPE_SOURCE=${INPUT_PROTOTYPE_SOURCE}"
  fi
  if [[ -n "$INPUT_BOUNDARY" ]]; then
    export_vars="${export_vars},INPUT_BOUNDARY=${INPUT_BOUNDARY}"
  fi
  if [[ -n "$INPUT_INLIERS" ]]; then
    export_vars="${export_vars},INPUT_INLIERS=${INPUT_INLIERS}"
  fi
  if [[ -n "$INPUT_X_OUTLIERS" ]]; then
    export_vars="${export_vars},INPUT_X_OUTLIERS=${INPUT_X_OUTLIERS}"
  fi
  if [[ -n "$INPUT_Y_OUTLIERS" ]]; then
    export_vars="${export_vars},INPUT_Y_OUTLIERS=${INPUT_Y_OUTLIERS}"
  fi

  if $DRY_RUN; then
    echo "[DRY RUN] sbatch --job-name=${job_name} --export=${export_vars} train_eoss.slurm"
  else
    echo "Submitting: ${job_name}"
    sbatch --job-name="${job_name}" --export="${export_vars}" "$SCRIPT_DIR/train_eoss.slurm"
  fi
}

echo "=============================================="
echo "DRY_RUN: $DRY_RUN (use --run to submit)"
echo "Fork descendants: ${#RUN_IDS_ARRAY[@]} baselines x ${#DROP_MULTS_ARRAY[@]} drops"
echo "=============================================="

for idx in "${!RUN_IDS_ARRAY[@]}"; do
  for drop_mult in "${DROP_MULTS_ARRAY[@]}"; do
    submit_job "${RUN_IDS_ARRAY[$idx]}" "${CONT_STEPS_ARRAY[$idx]}" "${BASE_LRS_ARRAY[$idx]}" "$drop_mult"
  done
done

if $DRY_RUN; then
  echo "This was a dry run. Use --run to actually submit jobs."
fi
