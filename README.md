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
sbatch --export=MODEL=cnn,OPTIMIZER=fullgd,LR=0.005,LMAX_DECAY=1 train_eoss.slurm
```

### Schedule Control

```bash
# Drop once when lambda_max crosses threshold (default when LMAX_DECAY=1)
sbatch --export=MODEL=cnn,OPTIMIZER=sgd,LR=0.01,LMAX_SCHEDULE=drop,LMAX_DROP_MULT=0.5 train_eoss.slurm

# Linear decay when lambda_max crosses threshold
sbatch --export=MODEL=cnn,OPTIMIZER=sgd,LR=0.01,LMAX_SCHEDULE=decay train_eoss.slurm
```

### Custom Grid (Loop Spec)

```bash
# Dry run (default)
./launch_ablation.sh --custom \
  --models "mlp cnn" \
  --optimizers "sgd adam" \
  --lrs "0.001 0.005" \
  --batches "8 32" \
  --lmax-schedule "none drop"

# Submit
./launch_ablation.sh --custom \
  --models "mlp cnn" \
  --optimizers "sgd adam" \
  --lrs "0.001 0.005" \
  --batches "8 32" \
  --lmax-schedule "none drop" \
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
2. **Steps auto-calculated**: 100/lr for full-GD, 500/lr for SGD
3. **LR schedule default**: `LMAX_DECAY=1` maps to `drop` unless `LMAX_SCHEDULE` is set
4. **Feature prototypes** require seed runs (already configured in `launch_ablation.sh`)

---

## Files

| File | Purpose |
|------|---------|
| `train_eoss.slurm` | Unified template (handles all optimizers) |
| `launch_ablation.sh` | Grid launcher with `--preset` and `--dry-run` |
| `prototype_registry.json` | Seed run IDs for feature tracking |
| `train_eoss_*.slurm` | Legacy per-optimizer scripts (still work) |
