#!/usr/bin/env python3
"""sweep_digest.py — walk a W&B project and emit a machine-readable digest.

For every run in the project:
  * fetch history for the core + any detected prototype-group metrics,
  * run every applicable primitive from `eos_signals`,
  * emit a flat dict of signal values (one row per run in digest.csv),
  * render a small annotated PNG per (run, metric) showing EoS threshold +
    first-crossing marker drawn from the same primitives the loop scored.

Output layout:
    <out-dir>/
        digest.json
        digest.csv
        digest_plots/<run_id>_<metric_safe>.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the repo root importable so we can use `eos_signals`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eos_signals import primitives  # noqa: E402


CORE_METRICS = ["full_loss", "lambda_max"]
GROUP_METRIC_SUFFIXES = ["full_loss", "lambda_max", "grad_hessian_grad", "grad_norm", "grad_vmax_cos2"]
HPARAM_KEYS = [
    "lr", "learning_rate", "optimizer", "batch_size", "model", "dataset",
    "lmax_schedule", "lmax_drop_mult", "input_prototypes_mode",
    "input_prototype_source", "num_data", "steps", "weight_decay", "seed",
]


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "metric"


def _discover_group_metrics(history_cols: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Return {group_name: {metric_suffix: full_col}} for prototype columns.

    Column pattern used by training.py:
      input_space_prototypes/<category>/<group>/<metric>
    We treat <group> as the display name.
    """
    groups: Dict[str, Dict[str, str]] = {}
    for col in history_cols:
        if not col.startswith("input_space_prototypes/"):
            continue
        parts = col.split("/")
        if len(parts) != 4:
            continue
        _, _category, group, metric = parts
        if metric not in GROUP_METRIC_SUFFIXES:
            continue
        groups.setdefault(group, {})[metric] = col
    return groups


def _resolve_eta(hparams: Dict[str, Any]) -> Optional[float]:
    for key in ("lr", "learning_rate", "optimizer_lr"):
        v = hparams.get(key)
        if v is None:
            continue
        try:
            vf = float(v)
            if vf > 0:
                return vf
        except (TypeError, ValueError):
            pass
    return None


def _run_global_signals(
    df: pd.DataFrame, step_col: str, eta: Optional[float], *, start_step: float
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "lambda_max" in df.columns and eta is not None:
        res = primitives.eos_threshold_crossing(
            df, step_col, "lambda_max", eta, adam=False, start_step=start_step
        )
        out["global.eos_crossing.lambda_max.step"] = res["step"]
        out["global.eos_crossing.lambda_max.value_at_step"] = res["value_at_step"]
        out["global.eos_crossing.lambda_max.threshold"] = res["threshold"]

        clean = df[[step_col, "lambda_max"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not clean.empty:
            smin = clean[step_col].min()
            smax = clean[step_col].max()
            mid = smin + (smax - smin) / 2.0
            slope = primitives.slope_in_window(df, step_col, "lambda_max", mid, smax)
            out["global.slope.lambda_max.slope"] = slope["slope"]
            out["global.slope.lambda_max.r2"] = slope["r2"]
            out["global.slope.lambda_max.n"] = slope["n"]
            out["global.slope.lambda_max.start_step"] = slope["start_step"]
            out["global.slope.lambda_max.end_step"] = slope["end_step"]

    if "full_loss" in df.columns:
        clean = df[[step_col, "full_loss"]].apply(pd.to_numeric, errors="coerce").dropna()
        if not clean.empty:
            last_val = float(clean["full_loss"].iloc[-1])
            tol = max(1e-6, abs(last_val) * 0.1) if math.isfinite(last_val) else 1e-3
            plat = primitives.plateau_duration(df, step_col, "full_loss", tol, min_length=10)
            out["global.plateau.full_loss.length"] = plat["length"]
            out["global.plateau.full_loss.start_step"] = plat["start_step"]
            out["global.plateau.full_loss.mean"] = plat["mean"]
            out["global.plateau.full_loss.tolerance"] = plat["tolerance"]
            out["global.final.full_loss"] = last_val
    return out


def _run_group_signals(
    df: pd.DataFrame,
    step_col: str,
    groups: Dict[str, Dict[str, str]],
    eta: Optional[float],
    *,
    start_step: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for group, cols in groups.items():
        prefix = f"group.{group}"
        # Per-group EoS crossing on lambda_max.
        if "lambda_max" in cols and eta is not None:
            res = primitives.eos_threshold_crossing(
                df, step_col, cols["lambda_max"], eta, adam=False, start_step=start_step
            )
            out[f"{prefix}.eos_crossing.lambda_max.step"] = res["step"]
            out[f"{prefix}.eos_crossing.lambda_max.value_at_step"] = res["value_at_step"]
        # Per-group stability ratio summary.
        if "grad_hessian_grad" in cols and "lambda_max" in cols:
            s = primitives.stability_ratio_summary(
                df, step_col, cols["grad_hessian_grad"], cols["lambda_max"], smooth_window=1
            )
            for stat in ("median", "mean", "max", "min", "last", "n"):
                out[f"{prefix}.stability_ratio.{stat}"] = s.get(stat)
        # Per-group full_loss final + slope in second half + first decline onset.
        if "full_loss" in cols:
            clean = df[[step_col, cols["full_loss"]]].apply(pd.to_numeric, errors="coerce").dropna()
            if not clean.empty:
                out[f"{prefix}.final.full_loss"] = float(clean[cols["full_loss"]].iloc[-1])
                smin = clean[step_col].min()
                smax = clean[step_col].max()
                mid = smin + (smax - smin) / 2.0
                sl = primitives.slope_in_window(df, step_col, cols["full_loss"], mid, smax)
                out[f"{prefix}.slope.full_loss.slope"] = sl["slope"]
                out[f"{prefix}.slope.full_loss.n"] = sl["n"]
                onset = primitives.first_window_with_negative_slope(
                    df, step_col, cols["full_loss"],
                    window=10, slope_threshold=0.0, min_r2=0.5, start_step=start_step,
                )
                out[f"{prefix}.loss_decline_onset.full_loss.step"] = onset["step"]
                out[f"{prefix}.loss_decline_onset.full_loss.slope"] = onset["slope"]
                out[f"{prefix}.loss_decline_onset.full_loss.r2"] = onset["r2"]
        # Per-group grad_vmax_cos2 first crossing (tests H03 leading indicator).
        if "grad_vmax_cos2" in cols:
            col = cols["grad_vmax_cos2"]
            clean = df[[step_col, col]].apply(pd.to_numeric, errors="coerce").dropna()
            if not clean.empty:
                # Threshold 0.1: 10% of total gradient energy aligned with v_1.
                cr = primitives.crossing_step(
                    df, step_col, col, threshold=0.1, direction="up", start_step=start_step,
                )
                out[f"{prefix}.cos_crossing.grad_vmax_cos2.step"] = cr["step"]
                out[f"{prefix}.cos_crossing.grad_vmax_cos2.value_at_step"] = cr["value_at_step"]
                out[f"{prefix}.cos_crossing.grad_vmax_cos2.threshold"] = cr["threshold"]
                out[f"{prefix}.final.grad_vmax_cos2"] = float(clean[col].iloc[-1])
                out[f"{prefix}.max.grad_vmax_cos2"] = float(clean[col].max())

    # Group separation across prototype groups.
    for metric_suffix in ("full_loss", "lambda_max"):
        group_frames: Dict[str, pd.DataFrame] = {}
        for group, cols in groups.items():
            if metric_suffix not in cols:
                continue
            col = cols[metric_suffix]
            sub = df[[step_col, col]].rename(columns={col: metric_suffix})
            group_frames[group] = sub
        if len(group_frames) >= 2:
            sep = primitives.group_separation(group_frames, step_col, metric_suffix)
            out[f"group_separation.{metric_suffix}.ratio"] = sep["ratio"]
            out[f"group_separation.{metric_suffix}.between_var"] = sep["between_var"]
            out[f"group_separation.{metric_suffix}.within_var"] = sep["within_var"]
    return out


def _render_plot(
    out_path: Path,
    df: pd.DataFrame,
    step_col: str,
    metric_col: str,
    title: str,
    threshold: Optional[float],
    crossing_step: Optional[float],
) -> None:
    clean = df[[step_col, metric_col]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(step_col)
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(clean[step_col], clean[metric_col], linewidth=1.2, label=metric_col)
    if threshold is not None and math.isfinite(threshold):
        ax.axhline(threshold, color="C3", linestyle="--", linewidth=0.9, label=f"threshold={threshold:.3g}")
    if crossing_step is not None and math.isfinite(crossing_step):
        ax.axvline(crossing_step, color="C2", linestyle=":", linewidth=0.9, label=f"crossing @ step {int(crossing_step)}")
    ax.set_xlabel(step_col)
    ax.set_ylabel(metric_col)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _fetch_history(run_obj, keys: List[str], samples: int) -> pd.DataFrame:
    try:
        return run_obj.history(samples=samples, keys=keys)
    except Exception as exc:
        print(f"  !! history fetch failed for {run_obj.id}: {exc}", file=sys.stderr)
        return pd.DataFrame()


def process_run(run_obj, step_col: str, samples: int, start_step: float,
                digest_plots: Path, project: str) -> Dict[str, Any]:
    summary_keys = list(run_obj.summary.keys()) if run_obj.summary else []
    groups = _discover_group_metrics(summary_keys)
    fetch_keys = [step_col, *CORE_METRICS]
    for cols in groups.values():
        fetch_keys.extend(cols.values())
    fetch_keys = list(dict.fromkeys(fetch_keys))

    df = _fetch_history(run_obj, fetch_keys, samples)
    hparams_full = dict(run_obj.config)
    hparams = {k: hparams_full.get(k) for k in HPARAM_KEYS}
    eta = _resolve_eta(hparams_full)

    signals: Dict[str, Any] = {}
    plots: List[str] = []

    if not df.empty:
        signals.update(_run_global_signals(df, step_col, eta, start_step=start_step))
        signals.update(_run_group_signals(df, step_col, groups, eta, start_step=start_step))

        # Render global lambda_max + full_loss plus one per prototype group.
        plot_targets: List[Tuple[str, Optional[float], Optional[float]]] = []
        thresh = primitives.eos_threshold(eta, adam=False) if eta else None
        for metric in CORE_METRICS:
            if metric not in df.columns:
                continue
            crossing = signals.get(f"global.eos_crossing.{metric}.step") if metric == "lambda_max" else None
            t = thresh if metric == "lambda_max" else None
            plot_targets.append((metric, t, crossing))
        for group, cols in groups.items():
            if "lambda_max" in cols:
                crossing = signals.get(f"group.{group}.eos_crossing.lambda_max.step")
                plot_targets.append((cols["lambda_max"], thresh, crossing))
        for metric, t, crossing in plot_targets:
            safe = _safe_filename(f"{run_obj.id}_{metric}")
            out_path = digest_plots / f"{safe}.png"
            _render_plot(out_path, df, step_col, metric, f"{run_obj.id} — {metric}", t, crossing)
            if out_path.exists():
                plots.append(str(out_path.relative_to(digest_plots.parent)))

    return {
        "run_id": run_obj.id,
        "run_name": run_obj.name,
        "project": project,
        "state": run_obj.state,
        "hparams": hparams,
        "eta": eta,
        "detected_groups": sorted(groups.keys()),
        "signals": signals,
        "plots": plots,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a digest.json + digest_plots/ for one W&B project.")
    p.add_argument("--project", required=True, help="W&B project (e.g. 'resnet-0.05').")
    p.add_argument("--entity", default=None, help="W&B entity/team if needed.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory for digest.json / digest.csv / digest_plots/. "
                        "Default: research-tick-results/ticks/<timestamp>/digests/<project>/")
    p.add_argument("--samples", type=int, default=5000, help="History samples per run.")
    p.add_argument("--step-col", default="_step", help="Step column (default: _step).")
    p.add_argument("--start-step", type=float, default=100.0,
                   help="Skip this many initial steps for crossing/slope primitives (default 100).")
    p.add_argument("--run-filter", default=None, help="Only include runs whose id or name matches this substring.")
    p.add_argument("--limit", type=int, default=None, help="Process at most N runs (for iterating fast).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import wandb

    if args.out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = _REPO_ROOT / "research-tick-results" / "ticks" / ts / "digests" / args.project
    args.out_dir.mkdir(parents=True, exist_ok=True)
    digest_plots = args.out_dir / "digest_plots"
    digest_plots.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = list(api.runs(project_path))
    if args.run_filter:
        runs = [r for r in runs if args.run_filter in r.id or args.run_filter in (r.name or "")]
    if args.limit is not None:
        runs = runs[:args.limit]
    print(f"[sweep_digest] {len(runs)} run(s) in {project_path}")

    records: List[Dict[str, Any]] = []
    for i, run_obj in enumerate(runs, 1):
        t0 = time.time()
        print(f"[{i}/{len(runs)}] {run_obj.id} {run_obj.name!r}", flush=True)
        try:
            rec = process_run(run_obj, args.step_col, args.samples, args.start_step,
                              digest_plots, args.project)
        except Exception as exc:
            print(f"  !! failed: {exc}", file=sys.stderr)
            rec = {"run_id": run_obj.id, "run_name": run_obj.name, "error": str(exc)}
        records.append(rec)
        dt = time.time() - t0
        sig_count = len(rec.get("signals", {}))
        plot_count = len(rec.get("plots", []))
        print(f"    signals={sig_count} plots={plot_count} t={dt:.1f}s")

    digest = {
        "schema_version": 1,
        "project": args.project,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_registry": "eos_signals/registry.yaml",
        "runs": records,
    }
    json_path = args.out_dir / "digest.json"
    with json_path.open("w") as f:
        json.dump(digest, f, indent=2, default=str)

    flat_rows: List[Dict[str, Any]] = []
    for r in records:
        row = {
            "run_id": r.get("run_id"),
            "run_name": r.get("run_name"),
            "state": r.get("state"),
            "eta": r.get("eta"),
            "detected_groups": ";".join(r.get("detected_groups", []) or []),
        }
        for hk, hv in (r.get("hparams") or {}).items():
            row[f"hparam.{hk}"] = hv
        for sk, sv in (r.get("signals") or {}).items():
            row[sk] = sv
        if "error" in r:
            row["error"] = r["error"]
        flat_rows.append(row)
    csv_path = args.out_dir / "digest.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    print(f"[sweep_digest] wrote {json_path}")
    print(f"[sweep_digest] wrote {csv_path}")
    print(f"[sweep_digest] plots -> {digest_plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
