#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=mlp-cifar10
#SBATCH --output=logs/%x-%j.out

# ---------- Hyperparameters from env (with defaults) ----------
LR=${LR:-0.01}
NUM_DATA=${NUM_DATA:-8192}
BATCH=${BATCH:-8}
STEPS=${STEPS:-150000}

echo "Running with: LR=${LR}, NUM_DATA=${NUM_DATA}, BATCH=${BATCH}, STEPS=${STEPS}"

# ---------- Environment setup ----------
module load miniforge/24.3.0-0   # adjust if needed
conda activate eoss

cd /home/anakhag/projects/edge-of-stochastic-stability

export WANDB_PROJECT=eoss_adam_mlp
export WANDB_DIR="/home/anakhag/projects/wandb_eoss_adam_mlp"

# ---------- Training command ----------
python training.py \
  --dataset cifar10 \
  --model mlp \
  --loss ce \
  --adam \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --steps "${STEPS}" \
  --num-data "${NUM_DATA}" \
  --init-scale 0.2 \
  --dataset-seed 111 \
  --init-seed 8312 \
  --stop-loss 0.00001 \
  --lambdamax \
  --batch-sharpness \
  --classes 1 9 \
  --track-input-prototypes \
  --track-feature-prototypes-from 20260108_1246_57_lr0.01000_b8