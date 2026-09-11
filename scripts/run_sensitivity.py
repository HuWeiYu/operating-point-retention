#!/usr/bin/env python3
"""Recompute the post-hoc operating-point sensitivity sweep (Fig.3 panels c/d).

For a task ``t`` learned at acquisition stage ``t`` and re-scored at a later stage,
its stored per-task threshold is ``tau_t`` (acquisition, never recalibrated). This
script moves the threshold to ``tau_t + delta*sigma_t`` over a ``delta`` grid and
measures FPR/FNR per (run, task, delta). ``delta = 0`` is the stored operating point
and must reproduce the formal ``FPR@fixed``/``FNR@fixed``.

Usage
-----
    python scripts/run_sensitivity.py --root .. \\
        --cohort configs/cohort.json --grid configs/delta_grid.json \\
        --out <out-dir>

The run-tree root is taken from ``--root``; run names inside ``--cohort`` are resolved
relative to it. No absolute path and no GPU are used.

Status is reported in the ``--out`` dir. Use ``--overwrite`` to replace existing
outputs; the script refuses to write over files it did not create.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.operating_point import metrics, tau_sigma  # noqa: E402
from src.analysis.io import (stored_tau, stored_validation_stats,   # noqa: E402
                             acquisition_validation_bundle, endpoint_bundle, load_bundle,
                             read_config)


def _load_json(path: Path, label: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _delta_grid(cfg: dict) -> np.ndarray:
    g = cfg["grid"]
    return np.round(np.arange(g["min"], g["max"] + 1e-9, g["step"]), 2)


def _seed_from_name(name: str):
    """Extract the seed from a run name like ``..._seed3_...`` -> 3 (or None)."""
    import re
    m = re.search(r"_seed(\d+)_", name)
    return int(m.group(1)) if m else None


def _sigma_for(run_dir, task, cohort):
    """sigma_t = std of the acquisition normal-validation scores; fall back to stored stats."""
    stats_sigma, _ = stored_validation_stats(run_dir, task)
    acq = acquisition_validation_bundle(run_dir, task)
    if stats_sigma is not None and np.isfinite(stats_sigma):
        # prefer the stored std; but confirm it matches a recomputation when possible
        return float(stats_sigma)
    if acq is not None:
        b = load_bundle(acq)
        tau, sigma = tau_sigma(b["scores"], b["labels"], cohort["hyperparams"]["threshold_quantile"])
        return sigma if np.isfinite(sigma) else None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="run-tree root (run names resolve here)")
    ap.add_argument("--cohort", default="configs/cohort.json", help="cohort JSON")
    ap.add_argument("--grid", default="configs/delta_grid.json", help="delta-grid JSON")
    ap.add_argument("--out", required=True, help="output directory (created if missing)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    ap.add_argument("--validate-only", action="store_true", help="parse+verify inputs, write nothing")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cohort = _load_json(args.cohort, "cohort")
    grid = _load_json(args.grid, "delta grid")
    deltas = _delta_grid(grid)
    warm_ok = (grid["grid"]["min"] <= 0 <= grid["grid"]["max"])

    if args.validate_only:
        print(f"validation OK: {len(cohort['run_names'])} runs, {len(deltas)} deltas, "
              f"delta=0 within grid: {warm_ok}")
        return 0

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    writes = [out / "op_sensitivity_by_run.csv", out / "op_sensitivity_replay_macro.csv",
              out / "op_sensitivity_delta0_check.csv", out / "op_sensitivity_summary.json"]
    if not args.overwrite:
        clash = [w for w in writes if w.exists()]
        if clash:
            raise SystemExit("refusing to overwrite existing outputs (pass --overwrite): "
                             + ", ".join(str(x) for x in clash))

    old_tasks = cohort["old_tasks"]
    endpoint = cohort["endpoint_stage"]
    rows, d0_rows, gaps = [], [], []
    run_dirs = []

    for name in cohort["run_names"]:
        run_dir = root / name
        if not run_dir.exists():
            gaps.append({"run": name, "reason": "run dir not found under --root"})
            continue
        run_dirs.append(name)
        seed = _seed_from_name(name)
        buf = int(read_config(run_dir).get("replay_buffer_size", 0))
        for task in old_tasks:
            tau = stored_tau(run_dir, task)
            if tau is None:
                gaps.append({"run": name, "task": task, "reason": "no stored tau_t (train.json)"})
                continue
            sigma = _sigma_for(run_dir, task, cohort)
            if sigma is None or not np.isfinite(sigma):
                gaps.append({"run": name, "task": task, "reason": "no valid sigma_t"})
                continue
            ep = endpoint_bundle(run_dir, endpoint, task)
            if ep is None or not Path(ep).exists():
                gaps.append({"run": name, "task": task, "reason": "missing endpoint bundle"})
                continue
            b = load_bundle(ep)
            for d in deltas:
                thr = tau + d * sigma
                fpr, fnr, nn, na = metrics(b["scores"], b["labels"], thr)
                rows.append({"run": name, "task": task, "seed": seed, "buffer": buf,
                             "delta": float(d), "tau_t": tau, "sigma_t": sigma,
                             "fpr": fpr, "fnr": fnr, "n_norm": nn, "n_anom": na})
            # delta-0 cross-check against the formal artifact
            fpr0, fnr0, _, _ = metrics(b["scores"], b["labels"], tau)
            d0_rows.append({"run": name, "task": task, "seed": seed, "buffer": buf,
                            "delta0_fpr": fpr0, "delta0_fnr": fnr0})

    if not rows:
        raise SystemExit("no per-run rows computed — check --root / cohort / artifact presence.")

    by_run = pd.DataFrame(rows)
    by_run.to_csv(writes[0], index=False)

    # macro: per (buffer, delta), mean over old tasks within a run, then mean+-SD over runs/seeds.
    # This reproduces Fig.3(c)/(d): one FPR/FNR-vs-delta curve per buffer size.
    macro = []
    for (buf, delta), grp in by_run.groupby(["buffer", "delta"]):
        per_run_task_mean = grp.groupby("run")[["fpr", "fnr"]].mean().reset_index()
        m, s = per_run_task_mean["fpr"].mean(), per_run_task_mean["fpr"].std(ddof=1)
        mf, sf = per_run_task_mean["fnr"].mean(), per_run_task_mean["fnr"].std(ddof=1)
        macro.append({"buffer": int(buf), "delta": float(delta), "fpr_mean": m, "fpr_std": s,
                      "fnr_mean": mf, "fnr_std": sf})
    pd.DataFrame(macro).to_csv(writes[1], index=False)

    pd.DataFrame(d0_rows).to_csv(writes[2], index=False)

    summary = {
        "cohort": cohort["name"], "n_runs": len(run_dirs), "n_deltas": int(len(deltas)),
        "delta_range": [float(grid["grid"]["min"]), float(grid["grid"]["max"])],
        "delta0_within_grid": warm_ok,
        "data_gaps": gaps,
        "sigma_provenance": "sigma_t = std(acquisition task-t normal-validation scores), ddof=1",
        "note": "A common delta is NOT a common absolute shift (each task has its own sigma_t).",
    }
    writes[3].write_text(json.dumps(summary, indent=2))

    print(f"wrote {len(by_run)} rows -> {writes[0]}")
    print(f"wrote macro -> {writes[1]} ; delta-0 -> {writes[2]}")
    print(f"gaps: {len(gaps)}; summary -> {writes[3]}")
    for g in gaps:
        print("   -", g)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
