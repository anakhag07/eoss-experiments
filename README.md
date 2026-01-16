# EoSS Experiments

## The Experiment Grid

| Ablation | Models | LRs | Other | Total Jobs |
|----------|--------|-----|-------|------------|
| **Full-GD + λ_max decay** | mlp, cnn, resnet | 0.0001, 0.001, 0.005, 0.01 | decay on/off | 24 |
| SGD baselines | mlp | 0.001, 0.005, 0.01, 0.05 | batch 8/32/128 | 12 |
| Adam | mlp | 0.001, 0.005, 0.01 | batch 8/32/64 | 9 |
| Momentum | mlp | 0.001, 0.005, 0.01 | batch 8/32/64 | 9 |

---

## Running Experiments

### Full-GD Grid (Primary)

```bash
# Preview what will run
./launch_ablation.sh --preset full_gd

# Submit all 24 jobs
./launch_ablation.sh --preset full_gd --run
```

### All Optimizer Ablations

```bash
./launch_ablation.sh --preset all --run
```

### Single Custom Job

```bash
sbatch --export=MODEL=cnn,OPTIMIZER=full_gd,LR=0.005,LMAX_DECAY=1 train_eoss.slurm
```

### Custom Grid (Loop Spec)

```bash
# Dry run (default)
./launch_ablation.sh --custom \
  --models "mlp cnn" \
  --optimizers "sgd adam" \
  --lrs "0.001 0.005" \
  --batches "8 32" \
  --lmax-decay "0 1"

# Submit
./launch_ablation.sh --custom \
  --models "mlp cnn" \
  --optimizers "sgd adam" \
  --lrs "0.001 0.005" \
  --batches "8 32" \
  --lmax-decay "0 1" \
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
./launch_ablation.sh --preset full_gd --run
# Submits 24 jobs with correct prototypes and auto-calculated steps
```

---

## Gotchas

1. **Batch sharpness hangs on full-GD** → automatically disabled
2. **Steps auto-calculated**: 100/lr for full-GD, 500/lr for SGD
3. **Feature prototypes** require seed runs (already configured in `launch_ablation.sh`)

---

## Files

| File | Purpose |
|------|---------|
| `train_eoss.slurm` | Unified template (handles all optimizers) |
| `launch_ablation.sh` | Grid launcher with `--preset` and `--dry-run` |
| `prototype_registry.json` | Seed run IDs for feature tracking |
| `train_eoss_*.slurm` | Legacy per-optimizer scripts (still work) |
