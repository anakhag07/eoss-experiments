# EoSS Experiments

## The Experiment Grid

| Ablation | Models | LRs | Other | Total Jobs |
|----------|--------|-----|-------|------------|
| **Full-GD + λ_max schedule** | mlp, cnn, resnet | 0.0001, 0.001, 0.005, 0.01 | drop/none | 24 |
| SGD baselines | mlp | 0.001, 0.005, 0.01, 0.05 | batch 8/32/128 | 12 |
| Adam | mlp | 0.001, 0.005, 0.01 | batch 8/32/64 | 9 |
| Momentum | mlp | 0.001, 0.005, 0.01 | batch 8/32/64 | 9 |

---

## Running Experiments

### Full-GD Grid (Primary)

```bash
# Preview what will run
./launch_ablation.sh --preset fullgd

# Submit all 24 jobs
./launch_ablation.sh --preset fullgd --run
```

### All Optimizer Ablations

```bash
./launch_ablation.sh --preset all --run
```

### Single Custom Job

```bash
sbatch --export=MODEL=cnn,OPTIMIZER=fullgd,LR=0.005,PROJECT_NAME=eoss-train-run,LMAX_SCHEDULE=drop train_eoss.slurm
```

### Config-First Job

`train_eoss.slurm` now launches `training.py` with a base JSON config and then applies env-driven CLI overrides on top.

```bash
sbatch --export=CONFIG_PATH=configs/your_base.json,MODEL=cnn,OPTIMIZER=sgd,LR=0.01 train_eoss.slurm
```

You can also pass the config through the ablation launcher:

```bash
./launch_ablation.sh --preset sgd --config-path configs/your_base.json --run
```

### Input Prototype Ablations

Use the launcher's modern input-prototype knobs to sweep mode, source, and per-class subset counts.

```bash
# Preview a validation-mode ablation over source and subset sizes
./launch_ablation.sh --custom \
  --models "mlp" \
  --optimizers "sgd" \
  --lrs "0.01" \
  --batches "128" \
  --input-prototypes-modes "val" \
  --input-prototype-sources "generate from:$PROTO_RUN" \
  --input-boundary-counts "5 10" \
  --input-inliers-counts "5 10" \
  --input-x-outlier-counts "5" \
  --input-y-outlier-counts "5"

# Turn prototypes off explicitly for a baseline run
./launch_ablation.sh --custom \
  --models "mlp" \
  --optimizers "sgd" \
  --lrs "0.01" \
  --input-prototype-sources "none"
```

Counts are only emitted when provided. Use `--input-prototype-sources "none"` for a no-prototype baseline, or `generate` / `from:<path-or-run>` together with one or more of `--input-boundary-counts`, `--input-inliers-counts`, `--input-x-outlier-counts`, and `--input-y-outlier-counts`.

### Schedule Control

```bash
# Drop once when lambda_max crosses threshold
sbatch --export=MODEL=cnn,OPTIMIZER=sgd,LR=0.01,LMAX_SCHEDULE=drop,LMAX_DROP_MULT=0.5 train_eoss.slurm
```

### Custom Grid (Loop Spec)

```bash
# Dry run (default)
./launch_ablation.sh --custom \
  --models "mlp cnn" \
  --optimizers "sgd adam" \
  --lrs "0.001 0.005" \
  --batches "8 32" \
  --lmax-schedule "none drop" \
  --project-name "eoss-train-with-outliers"

# Submit
./launch_ablation.sh --custom \
  --models "mlp cnn" \
  --optimizers "sgd adam" \
  --lrs "0.001 0.005" \
  --batches "8 32" \
  --lmax-schedule "none drop" \
  --project-name "eoss-train-with-outliers" \
  --run
```

---

## Before vs After

**Before** (one script per optimizer, manual prototype IDs):
```bash
# Had to edit train_eoss_full_gd.slurm to change MODEL
# Had to remember prototype run IDs
# Had to calculate STEPS manually (100/lr)
sbatch train_eoss_full_gd.slurm
sbatch train_eoss_full_gd.slurm  # edit file, submit again
sbatch train_eoss_full_gd.slurm  # edit file, submit again...
```

**After** (one command):
```bash
./launch_ablation.sh --preset fullgd --run
# Submits 24 jobs with correct prototypes and auto-calculated steps
```

---

## Gotchas

1. **Batch sharpness hangs on full-GD** → automatically disabled
2. **Steps auto-calculated**: 100/lr for full-GD, 200/lr for SGD/Adam/Momentum
3. **LR schedule choices**: launcher support is `LMAX_SCHEDULE=none|drop`
4. **Feature prototypes** require seed runs (already configured in `launch_ablation.sh`)
5. **Adam sharpness metric**: `train_eoss.slurm` always adds `--precond-lmax` when `OPTIMIZER=adam`
6. **Base config precedence**: `CONFIG_PATH` supplies defaults, but exported env vars still win because they are emitted as CLI overrides
7. **Input prototype launcher interface**: use `INPUT_PROTOTYPE_SOURCE` plus per-subset counts; legacy launcher flags are removed

---

## Files

| File | Purpose |
|------|---------|
| `train_eoss.slurm` | Unified template (handles all optimizers) |
| `launch_ablation.sh` | Grid launcher with `--preset` and `--dry-run` |
| `prototype_registry.json` | Seed run IDs for feature tracking |
| `train_eoss_*.slurm` | Legacy per-optimizer scripts (still work) |
