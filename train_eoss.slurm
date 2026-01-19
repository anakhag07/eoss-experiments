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
#   sbatch --export=MODEL=resnet,OPTIMIZER=full_gd,LR=0.005 train_eoss.slurm
# =============================================================================

# ---------- Hyperparameters from env (with defaults) ----------
MODEL=${MODEL:-mlp}
OPTIMIZER=${OPTIMIZER:-sgd}        # sgd, adam, momentum, full_gd
DATASET=${DATASET:-cifar10}
LOSS=${LOSS:-ce}
LR=${LR:-0.01}
BATCH=${BATCH:-8}
NUM_DATA=${NUM_DATA:-10000}
INIT_SCALE=${INIT_SCALE:-0.2}
STEPS=${STEPS:-}                   # Auto-calculated if empty
LMAX_DECAY=${LMAX_DECAY:-0}
CLASSES=${CLASSES:-"1 9"}
TRACK_FEATURE_PROTOTYPES_FROM=${TRACK_FEATURE_PROTOTYPES_FROM:-""}

# ---------- Auto-calculate steps if not provided ----------
if [[ -z "$STEPS" ]]; then
  if [[ "$OPTIMIZER" == "full_gd" ]]; then
    # Full-GD: 100/lr steps
    STEPS=$(awk -v lr="$LR" 'BEGIN { printf "%d", 100 / lr }')
  else
    # SGD/Adam/Momentum: 200/lr steps
    STEPS=$(awk -v lr="$LR" 'BEGIN { printf "%d", 200 / lr }')
  fi
fi

# ---------- Full-GD overrides batch to full dataset ----------
if [[ "$OPTIMIZER" == "full_gd" ]]; then
  BATCH="$NUM_DATA"
fi

echo "=============================================="
echo "MODEL=${MODEL}, OPTIMIZER=${OPTIMIZER}, DATASET=${DATASET}"
echo "LR=${LR}, BATCH=${BATCH}, NUM_DATA=${NUM_DATA}, STEPS=${STEPS}"
echo "LMAX_DECAY=${LMAX_DECAY}, CLASSES=${CLASSES}"
echo "=============================================="

# ---------- Environment setup ----------
module load miniforge/24.3.0-0
conda activate eoss

cd /home/anakhag/projects/edge-of-stochastic-stability

export WANDB_PROJECT="eoss-${OPTIMIZER}-${MODEL}-${LOSS}-${DATASET}"
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
  full_gd|sgd)
    # No extra flags for vanilla SGD or full-GD
    ;;
  *)
    echo "Unknown optimizer: $OPTIMIZER"
    exit 1
    ;;
esac

# ---------- Build optional flags ----------
OPTIONAL_FLAGS=""

# Batch sharpness: skip for full-GD (times out on large batches)
if [[ "$OPTIMIZER" != "full_gd" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --batch-sharpness"
fi

# LR decay triggered by λ_max
if [[ "$LMAX_DECAY" == "1" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --lmax-decay --lmax-decay-target-lr 0.000001"
fi

# Feature prototype tracking
if [[ -n "$TRACK_FEATURE_PROTOTYPES_FROM" ]]; then
  OPTIONAL_FLAGS="$OPTIONAL_FLAGS --track-feature-prototypes-from $TRACK_FEATURE_PROTOTYPES_FROM"
fi

# ---------- Training command ----------
export CMD = "python -u training.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --loss "$LOSS" \
  --batch "$BATCH" \
  --lr "$LR" \
  --steps "$STEPS" \
  --num-data "$NUM_DATA" \
  --init-scale "$INIT_SCALE" \
  --dataset-seed 111 \
  --init-seed 8312 \
  --stop-loss 0.00001 \
  --lambdamax \
  --classes $CLASSES \
  --track-input-prototypes \
  --rare-measure \
  $OPTIMIZER_FLAGS \
  $OPTIONAL_FLAGS"

  echo "Running: $CMD"
  eval $CMD
  