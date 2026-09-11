#!/usr/bin/env python3
"""Evaluate a trained run dir: stored-threshold FPR/FNR + image AUROC (offline, CPU).

Reads the *saved* artifacts of a ``train_continual.py`` run directory and recomputes the
audited per-task metrics the paper reports — FPR and FNR at the stored per-task threshold
``tau_t`` (Eq. 1, never recalibrated) plus image-level AUROC — for every old task and for
the final (current) task. It never re-runs the model and needs no GPU.

Usage
-----
    python scripts/evaluate.py --run-dir ./runs/shared_ft_replay10 --output-dir ./results/shared_ft_replay10

The run dir must be a complete ``train_continual.py`` output (it has ``config.json``,
``stage_SS/train.json``, ``stage_SS/task_tt*.npz`` bundles, ``validation_metrics.csv``).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.io import (  # noqa: E402
    read_config,
    load_bundle,
    stored_tau,
    stored_validation_stats,
    endpoint_bundle,
    load_stage_metrics,
)
from src.metrics.operating_point import metrics, image_auroc, tau_sigma  # noqa: E402

OLD_TASKS = (1, 2, 3, 4)


def _auroc_of(bundle) -> float:
    b = load_bundle(bundle)
    return float(image_auroc(b["scores"], b["labels"]))


def _stored_of(run_dir: Path, task: int, tail: int):
    """Row: stored-threshold FPR/FNR + AUROC for old task ``task`` at the endpoint."""
    tau = stored_tau(run_dir, task)
    sigma, q99 = stored_validation_stats(run_dir, task)
    bundle = endpoint_bundle(run_dir, tail, task)
    if tau is None or bundle is None or not Path(bundle).exists():
        return None
    b = load_bundle(bundle)
    fpr, fnr, nn, na = metrics(b["scores"], b["labels"], tau)
    return {
        "scope": "old",
        "task": task,
        "tau_t": tau,
        "sigma_t": sigma,
        "q99": q99,
        "FPR_at_tau": fpr,
        "FNR_at_tau": fnr,
        "image_AUROC": float(image_auroc(b["scores"], b["labels"])),
        "n_normal": nn,
        "n_anomaly": na,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="a train_continual.py output dir")
    ap.add_argument("--output-dir", required=True, help="directory for the metrics summary")
    ap.add_argument("--endpoint-stage", type=int, default=5,
                    help="final stream stage (default 5)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"--run-dir not found: {run_dir}")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "evaluation_metrics.csv"
    if csv_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {csv_path} (pass --overwrite)")

    config = read_config(run_dir)
    tail = args.endpoint_stage
    rows = []
    for task in OLD_TASKS:
        if task >= tail:
            break
        row = _stored_of(run_dir, task, tail)
        if row is not None:
            rows.append(row)
        else:
            print(f"[warn] old task {task}: missing tau_t or endpoint bundle; skipped", flush=True)

    # current (final) task image-AUROC from its own endpoint bundle, if present.
    cur_bundle = endpoint_bundle(run_dir, tail, tail)
    if cur_bundle is not None and Path(cur_bundle).exists():
        rows.append({
            "scope": "current", "task": tail, "tau_t": None, "sigma_t": None, "q99": None,
            "FPR_at_tau": None, "FNR_at_tau": None,
            "image_AUROC": _auroc_of(cur_bundle), "n_normal": None, "n_anomaly": None,
        })
    else:
        print("[warn] current task endpoint bundle not found; AUROC skipped", flush=True)

    if not rows:
        raise SystemExit("no evaluable tasks found in run dir — is it a complete training output?")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"evaluated run: {run_dir}")
    print(f"  method={config.get('method')} replay_buffer_size="
          f"{config.get('replay_buffer_size', 0)} endpoint_stage={tail}")
    for r in rows:
        scope = r["scope"]
        if scope == "old":
            print(f"  old task {r['task']}: FPR@tau={r['FPR_at_tau']:.4f} "
                  f"FNR@tau={r['FNR_at_tau']:.4f} AUROC={r['image_AUROC']:.4f}")
        else:
            print(f"  current task {r['task']}: AUROC={r['image_AUROC']:.4f}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
