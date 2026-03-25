#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT_DEFAULT="/home/anakhag/projects/eos/edge-of-stochastic-stability-and-memorization"

ABLAT_OUT=$(mktemp)
TRAIN_OUT=$(mktemp)
trap 'rm -f "$ABLAT_OUT" "$TRAIN_OUT"' EXIT

bash "$SCRIPT_DIR/launch_ablation.sh" --preset sgd --config-path configs/your_base.json > "$ABLAT_OUT"
grep -Eq -- '--export=.*CONFIG_PATH=configs/your_base.json' "$ABLAT_OUT"

CONFIG_PATH=configs/your_base.json \
REPO_ROOT="$REPO_ROOT_DEFAULT" \
SKIP_ENV_SETUP=1 \
PRINT_COMMAND_ONLY=1 \
MODEL=mlp OPTIMIZER=adam LR=0.01 BATCH=64 NUM_DATA=10000 \
bash "$SCRIPT_DIR/train_eoss.slurm" > "$TRAIN_OUT"

grep -Eq -- '--config configs/your_base.json' "$TRAIN_OUT"
grep -Eq -- '--adam --precond-lmax' "$TRAIN_OUT"
grep -Eq -- '--batch-sharpness' "$TRAIN_OUT"

echo "Wiring checks passed."
