#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local W&B runs and derive stability ratios.")
    parser.add_argument("--wandb-root", type=Path, required=True, help="Root directory containing W&B run folders or a project directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output file path (.csv or .json).")
    parser.add_argument("--include-history", action="store_true", help="Also inspect wandb-history.jsonl when summary metrics are missing.")
    return parser.parse_args()


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _read_history_last(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    last: Dict[str, object] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    last.update(payload)
    except OSError:
        return {}
    return last


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    candidates: List[Path] = []
    for path in sorted(root.rglob("wandb-summary.json")):
        candidates.append(path.parent)
    return candidates


def _load_run_record(run_dir: Path, include_history: bool) -> Dict[str, object]:
    summary = _read_json(run_dir / "files" / "wandb-summary.json")
    if summary is None:
        summary = _read_json(run_dir / "wandb-summary.json") or {}

    config = _read_json(run_dir / "files" / "config.json")
    if config is None:
        config = _read_json(run_dir / "config.json") or {}

    history = {}
    if include_history:
        history = _read_history_last(run_dir / "files" / "wandb-history.jsonl")
        if not history:
            history = _read_history_last(run_dir / "wandb-history.jsonl")

    merged: Dict[str, object] = {}
    merged.update(history)
    merged.update(summary)

    record: Dict[str, object] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
    }

    wandb_meta = _read_json(run_dir / "files" / "wandb-metadata.json") or _read_json(run_dir / "wandb-metadata.json") or {}
    if isinstance(wandb_meta, dict):
        record["program"] = wandb_meta.get("program")

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict) and "value" in value:
                record[f"config/{key}"] = value["value"]
            else:
                record[f"config/{key}"] = value

    for key, value in merged.items():
        record[key] = value

    _add_stability_ratio_columns(record)
    return record


def _add_stability_ratio_columns(record: Dict[str, object]) -> None:
    global_lambda = _safe_float(record.get("lambda_max"))
    if global_lambda and global_lambda != 0.0:
        global_grad_hessian_grad = _safe_float(record.get("grad_hessian_grad"))
        if global_grad_hessian_grad is not None:
            record["stability_ratio"] = global_grad_hessian_grad / global_lambda

    for key, value in list(record.items()):
        if not isinstance(key, str):
            continue
        if not key.endswith("grad_hessian_grad"):
            continue
        numerator = _safe_float(value)
        if numerator is None or global_lambda in (None, 0.0):
            continue
        prefix = key[: -len("grad_hessian_grad")]
        ratio_key = f"{prefix}stability_ratio"
        record[ratio_key] = numerator / global_lambda


def _sorted_fieldnames(records: List[Dict[str, object]]) -> List[str]:
    fieldnames = set()
    for record in records:
        fieldnames.update(record.keys())
    preferred = [
        "run_name",
        "run_dir",
        "config/model",
        "config/lr",
        "config/batch",
        "config/loss",
        "lambda_max",
        "grad_hessian_grad",
        "stability_ratio",
        "full_loss",
        "accuracy",
    ]
    ordered = [name for name in preferred if name in fieldnames]
    ordered.extend(sorted(fieldnames - set(ordered)))
    return ordered


def write_csv(records: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _sorted_fieldnames(records)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_json(records: List[Dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    run_dirs = list(_iter_run_dirs(args.wandb_root))
    records = [_load_run_record(run_dir, args.include_history) for run_dir in run_dirs]

    suffix = args.output.suffix.lower()
    if suffix == ".json":
        write_json(records, args.output)
    else:
        write_csv(records, args.output)

    print(f"Wrote {len(records)} run summaries to {args.output}")


if __name__ == "__main__":
    main()
