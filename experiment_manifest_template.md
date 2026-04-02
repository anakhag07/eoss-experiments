# Experiment Manifest

## Sweep Identity

- Date:
- Project name:
- Hypothesis label:
- Owner:

## Research Question

- What curvature-learning interaction am I testing?
- Which prototype groups matter for this sweep?

## Launch Surface

- Canonical command:

```bash
bash eoss_training_scripts/launch_ablation.sh --custom ...
```

- Dry-run checked: yes/no
- Submission command:

```bash
bash eoss_training_scripts/launch_ablation.sh --custom ... --run
```

## Variables Being Swept

- Model(s):
- Optimizer(s):
- Learning rate(s):
- Batch size(s):
- `LMAX_SCHEDULE` values:
- Prototype mode/source:
- Boundary counts:
- Inlier counts:
- X-outlier counts:
- Y-outlier counts:

## Fixed Controls

- Dataset:
- Loss:
- Classes:
- `NUM_DATA`:
- `INIT_SCALE`:
- Config path:

## Decision Metrics

- `lambda_max`
- subset `grad_hessian_grad`
- subset stability ratio = `grad_hessian_grad / lambda_max`
- cosine similarity metrics
- subset loss

## Expected Signal

- What pattern would count as support for the hypothesis?
- What pattern would falsify it?

## Queue Notes

- Max jobs intended in flight:
- Expected queue bottleneck:
- Polling cadence:

## Analysis Outputs

- W&B project directory:
- Summary command:

```bash
python eoss_training_scripts/summarize_wandb_runs.py --wandb-root <project-dir> --output runs.csv
```

- Output file(s):

## Decision Log

- Keep / narrow / discard:
- Main evidence:
- Next sweep:
