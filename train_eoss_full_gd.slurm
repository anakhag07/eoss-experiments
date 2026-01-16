#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=model-cifar10
#SBATCH --output=logs/%x-%j.out

# ---------- Hyperparameters from env (with defaults) ----------
LR=${LR:-0.01}
NUM_DATA=${NUM_DATA:-10000}
STEPS=${STEPS:-10000}
MODEL=${MODEL:-mlp}
LMAX_DECAY=${LMAX_DECAY:-0}
TRACK_FEATURE_PROTOTYPES_FROM=${TRACK_FEATURE_PROTOTYPES_FROM:-""}

echo "Running with: LR=${LR}, NUM_DATA=${NUM_DATA}, STEPS=${STEPS}, MODEL=${MODEL}, LMAX_DECAY=${LMAX_DECAY}"

# ---------- Environment setup ----------
module load miniforge/24.3.0-0   # adjust if needed
conda activate eoss

cd /home/anakhag/projects/edge-of-stochastic-stability

export WANDB_PROJECT=eoss-fullgd-${MODEL}-ce-cifar10
export WANDB_DIR="/home/anakhag/projects/eoss-fullgd-${MODEL}-ce-cifar10"

# ---------- Training command ----------
if [[ "${LMAX_DECAY}" == "1" ]]; then
  LMAX_DECAY_FLAG="--lmax-decay --lmax-decay-target-lr 0.000001"
else
  LMAX_DECAY_FLAG=""
fi

python training.py \
  --dataset cifar10 \
  --model "${MODEL}" \
  --loss ce \
  --batch "${NUM_DATA}" \
  --lr "${LR}" \
  --steps "${STEPS}" \
  --num-data "${NUM_DATA}" \
  --init-scale 0.2 \
  --dataset-seed 111 \
  --init-seed 8312 \
  --stop-loss 0.00001 \
  --lambdamax \
  --classes 1 9 \
  --track-input-prototypes \
  --track-feature-prototypes-from "${TRACK_FEATURE_PROTOTYPES_FROM}" \
  ${LMAX_DECAY_FLAG}
