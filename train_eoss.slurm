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
LMAX_SCHEDULE=${LMAX_SCHEDULE:-""} # none | decay | drop (empty => infer from LMAX_DECAY)
LMAX_DECAY=${LMAX_DECAY:-0}
LMAX_DROP_MULT=${LMAX_DROP_MULT:-0.5}
LMAX_DROP_TARGET_LR=${LMAX_DROP_TARGET_LR:-""}
CLASSES=${CLASSES:-"1 9"}
TRACK_FEATURE_PROTOTYPES_FROM=${TRACK_FEATURE_PROTOTYPES_FROM:-""}
PROJECT_NAME=${PROJECT_NAME:-""}

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
echo "LMAX_SCHEDULE=${LMAX_SCHEDULE}, LMAX_DECAY=${LMAX_DECAY}, CLASSES=${CLASSES}"
echo "=============================================="

# ---------- Environment setup ----------
module load miniforge/24.3.0-0
conda activate eoss

cd /home/anakhag/projects/edge-of-stochastic-stability

if [[ -n "$PROJECT_NAME" ]]; then
  export WANDB_PROJECT="$PROJECT_NAME"
else
  export WANDB_PROJECT="eoss-${OPTIMIZER}-${MODEL}-${LOSS}-${DATASET}"
fi
export WANDB_DIR="/home/anakhag/projects/${WANDB_PROJECT}"
mkdir -p "$WANDB_DIR"

# ---------- Build optimizer flags ----------
OPTIMIZER_FLAGS=""
case "$OPTIMIZER" in
  adam)
    OPTIMIZER_FLAGS="--adam"
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

# Lmax schedule selection (default to drop if LMAX_DECAY=1 and schedule unset)
EFFECTIVE_LMAX_SCHEDULE="$LMAX_SCHEDULE"
if [[ -z "$EFFECTIVE_LMAX_SCHEDULE" ]]; then
  if [[ "$LMAX_DECAY" == "1" ]]; then
    EFFECTIVE_LMAX_SCHEDULE="drop"
  else
    EFFECTIVE_LMAX_SCHEDULE="none"
  fi
fi

# Batch sharpness: skip for full-GD (times out on large batches)
if [[ "$OPTIMIZER" != "fullgd" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --batch-sharpness"
fi

# LR schedule triggered by λ_max
if [[ "$EFFECTIVE_LMAX_SCHEDULE" == "decay" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --lmax-decay --lmax-decay-target-lr 0.000001"
elif [[ "$EFFECTIVE_LMAX_SCHEDULE" == "drop" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --lmax-drop --lmax-drop-mult ${LMAX_DROP_MULT}"
  if [[ -n "$LMAX_DROP_TARGET_LR" ]]; then
    OPTIONAL_FLAGS="$OPTIONAL_FLAGS --lmax-drop-target-lr ${LMAX_DROP_TARGET_LR}"
  fi
fi

# Feature prototype tracking
if [[ -n "$TRACK_FEATURE_PROTOTYPES_FROM" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --track-feature-prototypes-from $TRACK_FEATURE_PROTOTYPES_FROM"
fi

# ---------- Training command ----------
CMD=(python -u training.py
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
  --classes $CLASSES
  --track-input-prototypes
  --rare-measure
  $OPTIMIZER_FLAGS
  $OPTIONAL_FLAGS
)
echo "Running: ${CMD[*]}"
"${CMD[@]}"
  
