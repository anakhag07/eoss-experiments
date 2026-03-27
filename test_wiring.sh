#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT_DEFAULT=$(realpath "$SCRIPT_DIR/../edge-of-stochastic-stability-and-memorization")

ABLAT_OUT=$(mktemp)
TRAIN_OUT=$(mktemp)
trap 'rm -f "$ABLAT_OUT" "$TRAIN_OUT"' EXIT

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

echo "Wiring checks passed."
