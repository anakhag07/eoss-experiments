#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --job-name=eoss-fork-dense
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-1

# =============================================================================
# Portable two-phase dense fork experiment.
#
# Phase 0 (array index 0): run a dense baseline.
# Phase 1 (array index 1): continue from a baseline run and drop LR at T*.
#
# Dense logging is always enabled with --dense-window DENSE_START DENSE_END
# DENSE_EVERY.
#
# Usage:
#   # Phase 1: baseline runs
#   for L in mse ce; do
#     for S in 1111 5555 7070; do
#       sbatch --array=0 --export=ALL,SEED=$S,LOSS=$L \
#         fork_intervention_proto_dense.slurm
#     done
#   done
#
#   # Phase 2: fork from a baseline
#   sbatch --array=1 --export=ALL,SEED=1111,LOSS=mse,BASELINE_RUN_ID=<id>,TSTAR_STEP=<step> \
#     fork_intervention_proto_dense.slurm
#
# Common overrides:
#   REPO_ROOT=/abs/path/to/repo
#   CONFIG_PATH=configs/your_base.json
#   WANDB_PROJECT=custom-project
#   RESULTS_DIR=/abs/path/to/results
#   PROTO_PATH=/abs/path/to/input_prototypes  # overrides the default package path
#   SKIP_ENV_SETUP=1
#   CONDA_ENV_NAME=eoss
#   MODULE_TO_LOAD=miniforge/24.3.0-0
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SEED=${SEED:-1234}
CLASSES=${CLASSES:-"1 9"}
LR=${LR:-0.01}
STEPS=${STEPS:-10000}
MODEL=${MODEL:-mlp}
LOSS=${LOSS:-mse}
OPTIMIZER=${OPTIMIZER:-fullgd}
DATASET=${DATASET:-cifar10_2cls}
NUM_DATA=${NUM_DATA:-10000}
INIT_SCALE=${INIT_SCALE:-0.2}
DATASET_SEED=${DATASET_SEED:-888}
DROP_MULT=${DROP_MULT:-0.5}
BASELINE_RUN_ID=${BASELINE_RUN_ID:-""}
TSTAR_STEP=${TSTAR_STEP:-""}
CONFIG_PATH=${CONFIG_PATH:-/home/anakhag/projects/eos/edge-of-stochastic-stability-and-memorization/configs/your_base.json}
REPO_ROOT=${REPO_ROOT:-/home/anakhag/projects/eos/edge-of-stochastic-stability-and-memorization}
RESULTS_DIR=${RESULTS_DIR:-""}
WANDB_DIR_BASE=${WANDB_DIR_BASE:-""}
PROTO_PATH=${PROTO_PATH:-/home/anakhag/projects/eos/eos_results/plaintext/cifar10_2cls_resnet/prototypes_package}
WANDB_PROJECT=${WANDB_PROJECT:-"fork-eoss-proto-dense-${OPTIMIZER}-${MODEL}-${LOSS}-${DATASET}"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
SYNC_WANDB=${SYNC_WANDB:-1}
SKIP_ENV_SETUP=${SKIP_ENV_SETUP:-0}
MODULE_TO_LOAD=${MODULE_TO_LOAD:-""}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-eoss}
PRINT_COMMAND_ONLY=${PRINT_COMMAND_ONLY:-0}
OPTIONAL_FLAGS=${OPTIONAL_FLAGS:-""}

INPUT_PROTOTYPES_MODE=${INPUT_PROTOTYPES_MODE:-train}
INPUT_BOUNDARY=${INPUT_BOUNDARY:-25}
INPUT_INLIERS=${INPUT_INLIERS:-25}
INPUT_X_OUTLIERS=${INPUT_X_OUTLIERS:-25}
INPUT_Y_OUTLIERS=${INPUT_Y_OUTLIERS:-25}

DENSE_START=${DENSE_START:-0}
DENSE_END=${DENSE_END:-$STEPS}
DENSE_EVERY=${DENSE_EVERY:-32}

append_split_args() {
  local raw="$1"
  if [[ -n "$raw" ]]; then
    # Intentional word splitting for shell-style extra CLI flags.
    # shellcheck disable=SC2206
    local split_args=( $raw )
    CMD+=("${split_args[@]}")
  fi
}

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$REPO_ROOT/$CONFIG_PATH"
fi

TRAINING_PY="$REPO_ROOT/training.py"

if [[ -z "$PROTO_PATH" ]]; then
  echo "PROTO_PATH is required and must point to the input prototype directory"
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -f "$TRAINING_PY" ]]; then
  echo "Training entrypoint not found: $TRAINING_PY"
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ ! -d "$PROTO_PATH" ]]; then
  echo "Prototype directory not found: $PROTO_PATH"
  exit 1
fi

export WANDB_PROJECT
if [[ -n "$WANDB_ENTITY" ]]; then
  export WANDB_ENTITY
fi
if [[ -n "$RESULTS_DIR" && -z "$WANDB_DIR_BASE" ]]; then
  WANDB_DIR_BASE="$RESULTS_DIR/wandb"
fi
WANDB_DIR_BASE=${WANDB_DIR_BASE:-/home/anakhag/projects/eos}
export WANDB_DIR="$WANDB_DIR_BASE/$WANDB_PROJECT"
export RESULTS="$WANDB_DIR"
mkdir -p "$WANDB_DIR"
echo "REPO_ROOT=${REPO_ROOT}"
echo "TRAINING_PY=${TRAINING_PY}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "PROTO_PATH=${PROTO_PATH}"
echo "INIT_SCALE=${INIT_SCALE}"
echo "WANDB_DIR=${WANDB_DIR}"
echo "RESULTS=${RESULTS}"
echo "PWD=$(pwd)"

if [[ "$OPTIMIZER" == "fullgd" ]]; then
  BATCH=${BATCH:-$NUM_DATA}
else
  BATCH=${BATCH:-128}
fi

CMD=(python -u "$TRAINING_PY"
  --config "$CONFIG_PATH"
  --dataset "$DATASET"
  --classes $CLASSES
  --num-data "$NUM_DATA"
  --model "$MODEL"
  --loss "$LOSS"
  --batch "$BATCH"
  --init-scale "$INIT_SCALE"
  --dataset-seed "$DATASET_SEED"
  --init-seed "$SEED"
  --steps "$STEPS"
  --lr "$LR"
  --lambdamax
  --input-prototypes-mode "$INPUT_PROTOTYPES_MODE"
  --input-prototype-source "from:$PROTO_PATH"
  --input-boundary "$INPUT_BOUNDARY"
  --input-inliers "$INPUT_INLIERS"
  --input-x-outliers "$INPUT_X_OUTLIERS"
  --input-y-outliers "$INPUT_Y_OUTLIERS"
  --dense-window "$DENSE_START" "$DENSE_END" "$DENSE_EVERY"
)

case "$OPTIMIZER" in
  adam)
    CMD+=(--adam --precond-lmax)
    ;;
  momentum)
    CMD+=(--momentum 0.9 --batch-sharpness)
    ;;
  sgd)
    CMD+=(--batch-sharpness)
    ;;
  fullgd)
    ;;
  *)
    echo "Unknown optimizer: $OPTIMIZER"
    exit 1
    ;;
esac

append_split_args "$OPTIONAL_FLAGS"

IDX=${SLURM_ARRAY_TASK_ID:-0}

case "$IDX" in
  0)
    NAME=${WANDB_NAME:-"fork_baseline_seed${SEED}_lr${LR}"}

    echo "=============================================="
    echo "DENSE PHASE 1: BASELINE"
    echo "SEED=${SEED} LOSS=${LOSS} OPTIMIZER=${OPTIMIZER}"
    echo "STEPS=${STEPS} dense-window=${DENSE_START}..${DENSE_END} every ${DENSE_EVERY}"
    echo "WANDB_PROJECT=${WANDB_PROJECT}"
    echo "=============================================="

    CMD+=(--wandb-name "$NAME")
    ;;

  1)
    if [[ -z "$BASELINE_RUN_ID" || -z "$TSTAR_STEP" ]]; then
      echo "Phase 2 requires BASELINE_RUN_ID and TSTAR_STEP"
      exit 1
    fi

    # CONT_STEP must be strictly < TSTAR_STEP so the training loop passes
    # through step==TSTAR_STEP when the scheduled LR drop check fires.
    # If cont_step == lr_drop_at_step, step_number is incremented past it
    # before the check, so the drop never triggers.
    CONT_STEP=${CONT_STEP:-$((TSTAR_STEP - 50))}
    if (( CONT_STEP >= TSTAR_STEP )); then
      echo "ERROR: CONT_STEP ($CONT_STEP) must be < TSTAR_STEP ($TSTAR_STEP)."
      exit 1
    fi

    NEW_LR=$(awk -v lr="$LR" -v m="$DROP_MULT" 'BEGIN { printf "%.10g", lr * m }')
    NAME=${WANDB_NAME:-"fork_exit_eos_seed${SEED}_lr${LR}_drop${DROP_MULT}_from_step${TSTAR_STEP}"}

    echo "=============================================="
    echo "DENSE PHASE 2: FORK"
    echo "SEED=${SEED} LOSS=${LOSS} OPTIMIZER=${OPTIMIZER}"
    echo "LR=${LR} -> ${NEW_LR} DROP=${DROP_MULT}"
    echo "BASELINE=${BASELINE_RUN_ID} T*=${TSTAR_STEP} CONT_STEP=${CONT_STEP}"
    echo "STEPS=${STEPS} dense-window=${DENSE_START}..${DENSE_END} every ${DENSE_EVERY}"
    echo "=============================================="

    CMD+=(
      --lr-drop-at-step "$TSTAR_STEP"
      --lr-drop-to "$NEW_LR"
      --cont-run-id "$BASELINE_RUN_ID"
      --cont-step "$CONT_STEP"
      --wandb-name "$NAME"
    )
    ;;

  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=$IDX (expected 0 or 1)"
    exit 1
    ;;
esac

echo "Running: ${CMD[*]}"
if [[ "$PRINT_COMMAND_ONLY" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"

if [[ "$SYNC_WANDB" == "1" ]]; then
  (
    cd "$WANDB_DIR"
    wandb sync --sync-all --include-offline --mark-synced --no-include-synced
  )
fi
