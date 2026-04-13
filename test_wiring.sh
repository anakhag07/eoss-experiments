#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT_DEFAULT=$(realpath "$SCRIPT_DIR/../edge-of-stochastic-stability-and-memorization")

ABLAT_OUT=$(mktemp)
TRAIN_OUT=$(mktemp)
FORK_OUT=$(mktemp)
trap 'rm -f "$ABLAT_OUT" "$TRAIN_OUT" "$FORK_OUT"' EXIT

bash "$SCRIPT_DIR/launch_ablation.sh" --preset sgd --config-path configs/your_base.json > "$ABLAT_OUT"
grep -Eq -- '--export=.*CONFIG_PATH=configs/your_base.json' "$ABLAT_OUT"

bash "$SCRIPT_DIR/launch_ablation.sh" --custom \
  --config-path configs/your_base.json \
  --lmax-schedule "drop" \
  --input-prototypes-modes "val" \
  --input-prototype-sources "generate" \
  --input-boundary-counts "5" \
  --input-inliers-counts "4" \
  --input-x-outlier-counts "3" \
  --input-y-outlier-counts "2" > "$ABLAT_OUT"

grep -Eq -- 'INPUT_PROTOTYPES_MODE=val' "$ABLAT_OUT"
grep -Eq -- 'LMAX_SCHEDULE=drop' "$ABLAT_OUT"
grep -Eq -- 'INPUT_PROTOTYPE_SOURCE=generate' "$ABLAT_OUT"
grep -Eq -- 'INPUT_BOUNDARY=5' "$ABLAT_OUT"
grep -Eq -- 'INPUT_INLIERS=4' "$ABLAT_OUT"
grep -Eq -- 'INPUT_X_OUTLIERS=3' "$ABLAT_OUT"
grep -Eq -- 'INPUT_Y_OUTLIERS=2' "$ABLAT_OUT"

CONFIG_PATH=configs/your_base.json \
REPO_ROOT="$REPO_ROOT_DEFAULT" \
SKIP_ENV_SETUP=1 \
PRINT_COMMAND_ONLY=1 \
LMAX_SCHEDULE=drop \
INPUT_PROTOTYPES_MODE=val \
INPUT_PROTOTYPE_SOURCE=generate \
INPUT_BOUNDARY=5 \
INPUT_INLIERS=4 \
INPUT_X_OUTLIERS=3 \
INPUT_Y_OUTLIERS=2 \
MODEL=mlp OPTIMIZER=adam LR=0.01 BATCH=64 NUM_DATA=10000 \
bash "$SCRIPT_DIR/train_eoss.slurm" > "$TRAIN_OUT"

grep -Eq -- '--config configs/your_base.json' "$TRAIN_OUT"
grep -Eq -- '--adam --precond-lmax' "$TRAIN_OUT"
grep -Eq -- '--batch-sharpness' "$TRAIN_OUT"
grep -Eq -- '--lmax-drop --lmax-drop-mult 0.5' "$TRAIN_OUT"
grep -Eq -- '--input-prototypes-mode val' "$TRAIN_OUT"
grep -Eq -- '--input-prototype-source generate' "$TRAIN_OUT"
grep -Eq -- '--input-boundary 5' "$TRAIN_OUT"
grep -Eq -- '--input-inliers 4' "$TRAIN_OUT"
grep -Eq -- '--input-x-outliers 3' "$TRAIN_OUT"
grep -Eq -- '--input-y-outliers 2' "$TRAIN_OUT"

CONFIG_PATH=configs/your_base.json \
REPO_ROOT="$REPO_ROOT_DEFAULT" \
SKIP_ENV_SETUP=1 \
PRINT_COMMAND_ONLY=1 \
MODEL=mlp OPTIMIZER=fullgd LR=0.01 NUM_DATA=10000 \
CONT_RUN_ID=abc123 CONT_STEP=42000 \
LR_DROP_AT_STEP=42000 LR_DROP_TO=0.005 \
CHECKPOINT_EVERY=500 WANDB_TAG=fork-test WANDB_NAME=fork_desc \
bash "$SCRIPT_DIR/train_eoss.slurm" > "$TRAIN_OUT"

grep -Eq -- '--cont-run-id abc123 --cont-step 42000' "$TRAIN_OUT"
grep -Eq -- '--lr-drop-at-step 42000 --lr-drop-to 0.005' "$TRAIN_OUT"
grep -Eq -- '--checkpoint-every 500' "$TRAIN_OUT"
grep -Eq -- '--wandb-tag fork-test' "$TRAIN_OUT"
grep -Eq -- '--wandb-name fork_desc' "$TRAIN_OUT"

bash "$SCRIPT_DIR/launch_fork_ablation.sh" \
  --baseline-run-ids "abc123 def456" \
  --cont-steps "42000 31000" \
  --base-lrs "0.01 0.005" \
  --drop-mults "0.5 0.2" \
  --config-path configs/your_base.json \
  --checkpoint-every 500 \
  --project-name fork-project \
  --input-prototypes-mode train \
  --input-prototype-source from:/tmp/proto \
  --input-boundary 25 \
  --input-inliers 25 \
  --input-x-outliers 25 \
  --input-y-outliers 25 > "$FORK_OUT"

grep -Eq -- 'CONT_RUN_ID=abc123,CONT_STEP=42000,LR_DROP_AT_STEP=42000,LR_DROP_TO=0.005' "$FORK_OUT"
grep -Eq -- 'CONT_RUN_ID=def456,CONT_STEP=31000,LR_DROP_AT_STEP=31000,LR_DROP_TO=0.001' "$FORK_OUT"
grep -Eq -- 'INPUT_PROTOTYPES_MODE=train,INPUT_PROTOTYPE_SOURCE=from:/tmp/proto' "$FORK_OUT"
grep -Eq -- 'INPUT_BOUNDARY=25,INPUT_INLIERS=25,INPUT_X_OUTLIERS=25,INPUT_Y_OUTLIERS=25' "$FORK_OUT"

echo "Wiring checks passed."
