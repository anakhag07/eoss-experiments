#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=eoss-train
#SBATCH --output=logs/%x-%j.out

# =============================================================================
# UNIFIED EOSS TRAINING SCRIPT
# =============================================================================
# Usage:
#   sbatch --export=MODEL=mlp,OPTIMIZER=sgd,LR=0.01 train_eoss.slurm
#   sbatch --export=MODEL=cnn,OPTIMIZER=adam,BATCH=64 train_eoss.slurm
#   sbatch --export=MODEL=resnet,OPTIMIZER=fullgd,LR=0.005 train_eoss.slurm
#   sbatch --export=PROJECT_NAME=my-project train_eoss.slurm
#   sbatch --export=CONFIG_PATH=configs/your_base.json,MODEL=mlp train_eoss.slurm
# =============================================================================

# ---------- Hyperparameters from env (with defaults) ----------
MODEL=${MODEL:-mlp}
OPTIMIZER=${OPTIMIZER:-sgd}        # sgd, adam, momentum, fullgd
DATASET=${DATASET:-cifar10}
LOSS=${LOSS:-ce}
LR=${LR:-0.01}
BATCH=${BATCH:-128}
NUM_DATA=${NUM_DATA:-10000}
INIT_SCALE=${INIT_SCALE:-0.2}
STEPS=${STEPS:-}                   # Auto-calculated if empty
LMAX_SCHEDULE=${LMAX_SCHEDULE:-"none"} # none | drop
LMAX_DROP_MULT=${LMAX_DROP_MULT:-0.5}
LMAX_DROP_TARGET_LR=${LMAX_DROP_TARGET_LR:-""}
CLASSES=${CLASSES:-"1 9"}
TRACK_FEATURE_PROTOTYPES_FROM=${TRACK_FEATURE_PROTOTYPES_FROM:-""}
PROJECT_NAME=${PROJECT_NAME:-""}
INPUT_PROTOTYPES_MODE=${INPUT_PROTOTYPES_MODE:-"train"}  # train | val
INPUT_PROTOTYPE_SOURCE=${INPUT_PROTOTYPE_SOURCE:-""}
INPUT_BOUNDARY=${INPUT_BOUNDARY:-""}
INPUT_INLIERS=${INPUT_INLIERS:-""}
INPUT_X_OUTLIERS=${INPUT_X_OUTLIERS:-""}
INPUT_Y_OUTLIERS=${INPUT_Y_OUTLIERS:-""}
CONFIG_PATH=${CONFIG_PATH:-configs/your_base.json}
REPO_ROOT=${REPO_ROOT:-/home/anakhag/projects/eos/edge-of-stochastic-stability-and-memorization}
SKIP_ENV_SETUP=${SKIP_ENV_SETUP:-0}
PRINT_COMMAND_ONLY=${PRINT_COMMAND_ONLY:-0}

# ---------- Auto-calculate steps if not provided ----------
if [[ -z "$STEPS" ]]; then
  if [[ "$OPTIMIZER" == "fullgd" ]] && [[ "$LOSS" == "ce" ]]; then
    # Full-GD: 100/lr steps
    STEPS=$(awk -v lr="$LR" 'BEGIN { printf "%d", 100 / lr }')
  elif [[ "$OPTIMIZER" == "fullgd" ]] && [[ "$LOSS" == "mse" ]]; then
    # Full-GD with MSE loss: 200/lr steps
    STEPS=$(awk -v lr="$LR" 'BEGIN { printf "%d", 200 / lr }')
  else
    # SGD/Adam/Momentum: 200/lr steps
    STEPS=$(awk -v lr="$LR" 'BEGIN { printf "%d", 200 / lr }')
  fi
fi

# ---------- Full-GD overrides batch to full dataset ----------
if [[ "$OPTIMIZER" == "fullgd" ]]; then
  BATCH="$NUM_DATA"
fi

echo "=============================================="
echo "MODEL=${MODEL}, OPTIMIZER=${OPTIMIZER}, DATASET=${DATASET}"
echo "LR=${LR}, BATCH=${BATCH}, NUM_DATA=${NUM_DATA}, STEPS=${STEPS}"
echo "LMAX_SCHEDULE=${LMAX_SCHEDULE}, CLASSES=${CLASSES}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "=============================================="

# ---------- Environment setup ----------
if [[ "$SKIP_ENV_SETUP" != "1" ]]; then
  module load miniforge/24.3.0-0
  conda activate eoss
fi

cd "$REPO_ROOT"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

if [[ -n "$PROJECT_NAME" ]]; then
  export WANDB_PROJECT="$PROJECT_NAME"
else
  export WANDB_PROJECT="eoss-${OPTIMIZER}-${MODEL}-${LOSS}-${DATASET}"
fi
export WANDB_DIR="/home/anakhag/projects/eos/${WANDB_PROJECT}"
mkdir -p "$WANDB_DIR"

# ---------- Build optimizer flags ----------
OPTIMIZER_FLAGS=""
case "$OPTIMIZER" in
  adam)
    OPTIMIZER_FLAGS="--adam --precond-lmax"
    ;;
  momentum)
    OPTIMIZER_FLAGS="--momentum 0.9"
    ;;
  fullgd|sgd)
    # No extra flags for vanilla SGD or full-GD
    ;;
  *)
    echo "Unknown optimizer: $OPTIMIZER"
    exit 1
    ;;
esac

# ---------- Build optional flags ----------
# Start with any user-provided OPTIONAL_FLAGS from the environment, then append.
OPTIONAL_FLAGS="${OPTIONAL_FLAGS:-}"

append_split_args() {
  local raw="$1"
  if [[ -n "$raw" ]]; then
    # Intentional word splitting for shell-like override strings.
    # shellcheck disable=SC2206
    local split_args=( $raw )
    CMD+=("${split_args[@]}")
  fi
}

# Batch sharpness: skip for full-GD (times out on large batches)
if [[ "$OPTIMIZER" != "fullgd" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --batch-sharpness"
fi

# LR schedule triggered by λ_max
if [[ "$LMAX_SCHEDULE" == "drop" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --lmax-drop --lmax-drop-mult ${LMAX_DROP_MULT}"
  if [[ -n "$LMAX_DROP_TARGET_LR" ]]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --lmax-drop-target-lr ${LMAX_DROP_TARGET_LR}"
  fi
elif [[ "$LMAX_SCHEDULE" != "none" ]]; then
  echo "Unsupported LMAX_SCHEDULE: $LMAX_SCHEDULE (expected 'none' or 'drop')"
  exit 1
fi

# Feature prototype tracking
if [[ -n "$TRACK_FEATURE_PROTOTYPES_FROM" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --track-feature-prototypes-from $TRACK_FEATURE_PROTOTYPES_FROM"
fi

# Input-space prototype configuration
if [[ -n "$INPUT_PROTOTYPES_MODE" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --input-prototypes-mode $INPUT_PROTOTYPES_MODE"
fi
if [[ -n "$INPUT_PROTOTYPE_SOURCE" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --input-prototype-source $INPUT_PROTOTYPE_SOURCE"
fi
if [[ -n "$INPUT_BOUNDARY" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --input-boundary $INPUT_BOUNDARY"
fi
if [[ -n "$INPUT_INLIERS" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --input-inliers $INPUT_INLIERS"
fi
if [[ -n "$INPUT_X_OUTLIERS" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --input-x-outliers $INPUT_X_OUTLIERS"
fi
if [[ -n "$INPUT_Y_OUTLIERS" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --input-y-outliers $INPUT_Y_OUTLIERS"
fi

# ---------- Training command ----------
read -r -a CLASSES_ARRAY <<< "$CLASSES"

CMD=(python -u training.py
  --config "$CONFIG_PATH"
  --dataset "$DATASET"
  --model "$MODEL"
  --loss "$LOSS"
  --batch "$BATCH"
  --lr "$LR"
  --steps "$STEPS"
  --num-data "$NUM_DATA"
  --init-scale "$INIT_SCALE"
  --dataset-seed 111
  --init-seed 8312
  --stop-loss 0.00001
  --lambdamax
)
if [[ ${#CLASSES_ARRAY[@]} -gt 0 ]]; then
  CMD+=(--classes "${CLASSES_ARRAY[@]}")
fi
append_split_args "$OPTIMIZER_FLAGS"
append_split_args "$OPTIONAL_FLAGS"

echo "Running: ${CMD[*]}"
if [[ "$PRINT_COMMAND_ONLY" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
  
