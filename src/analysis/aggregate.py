"""Aggregate per-run metrics the way the paper's replay figure does.

The convention (matching ``replay_multiseed_analysis.py`` and the paper):
  * a run's *old-task* metric is the mean over old tasks ``1..k-1`` at the final
    (endpoint) stage ``k``;
  * a *current-task* metric is the value at the final stage for task ``k`` (on a
    5-stage stream, task 5);
  * across seeds the report is ``mean +/- sample SD`` over the matched seeds;
  * routing is read as ``oracle`` when the run has oracle rows, else ``shared``;
  * a run is *skipped* (never zero-filled) when its artifacts are missing.

This module is pure pandas/NumPy and takes explicit paths. It never writes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from .io import load_run_dir


def _selected_route(stage) -> str | None:
    """'oracle' if the run exposes oracle rows, else 'shared', else None."""
    route_values = set(stage.get("route", pd.Series(dtype=str)).astype(str).str.lower())
    if "oracle" in route_values:
        return "oracle"
    if "shared" in route_values:
        return "shared"
    return None


def _at_stage(stage, task, stage_no):
    """Last row of ``task`` at ``stage_no`` (the endpoint row for that task)."""
    rows = stage[(stage["eval_task"] == task) & (stage["stage"] == stage_no)]
    return rows.iloc[-1] if not rows.empty else None


def per_run_metrics(run_dir) -> dict | None:
    """One run's aggregate (old-task means + current task + drift). ``None`` if incomplete.

    Fields mirror the paper's replay table: ``old_fpr_fixed_mean`` (over old tasks),
    ``old_current_i_auroc_mean``, ``old_acquisition_i_auroc_mean``,
    ``current_task_i_auroc``, ``old_normalized_score_drift_mean``, plus provenance.
    """
    run = load_run_dir(run_dir)
    if run is None:
        return None
    stage, drift = run["stage"], run["drift"]
    route = _selected_route(stage)
    if route and "route" in stage:
        stage = stage[stage["route"].astype(str).str.lower() == route]
    if route and "route" in drift:
        drift = drift[drift["route"].astype(str).str.lower() == route]

    final_stage = float(stage["stage"].max())
    tasks = sorted(int(t) for t in stage["eval_task"].dropna().unique())
    old_tasks = [t for t in tasks if t < final_stage]
    current_task = max(tasks) if tasks else None

    old_fpr, old_cur_auc, old_acq_auc, old_drift = [], [], [], []
    for t in old_tasks:
        row = _at_stage(stage, t, final_stage)
        if row is None:
            continue
        old_fpr.append(float(row["FPR@fixed"]))
        old_cur_auc.append(float(row["I-AUROC"]))
        acq = stage[(stage["eval_task"] == t) & (stage["stage"] == t)]
        if not acq.empty:
            old_acq_auc.append(float(acq.iloc[-1]["I-AUROC"]))
        d = drift[(drift["eval_task"] == t) & (drift["stage"] == final_stage)]
        if not d.empty and "normalized_score_drift" in d.columns:
            val = d.iloc[-1]["normalized_score_drift"]
            if val is not None and not np.isnan(val):
                old_drift.append(float(val))

    cur_row = _at_stage(stage, current_task, final_stage) if current_task is not None else None

    def _mean(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "method": run["method"],
        "buffer_size": run["buffer"],
        "seed": run["seed"],
        "n_old_tasks": len(old_fpr),
        "old_fpr_fixed_mean": _mean(old_fpr),
        "old_fpr_fixed_max": float(np.max(old_fpr)) if old_fpr else None,
        "old_current_i_auroc_mean": _mean(old_cur_auc),
        "old_acquisition_i_auroc_mean": _mean(old_acq_auc),
        "current_task_i_auroc": (float(cur_row["I-AUROC"]) if cur_row is not None else None),
        "old_normalized_score_drift_mean": _mean(old_drift),
    }


def collect_runs(run_dirs) -> tuple[list, list]:
    """(completed_rows, skipped) for a list of run directories."""
    rows, skipped = [], []
    for rd in run_dirs:
        r = per_run_metrics(rd)
        if r is None:
            skipped.append({"run_dir": str(Path(rd).resolve()),
                            "reason": "missing stage_metrics.csv or drift_metrics.csv"})
            continue
        r["run"] = Path(rd).name
        rows.append(r)
    return rows, skipped


def mean_std(xs) -> tuple[float, float | None]:
    """Mean and sample SD (ddof=1) across a list of numbers; SD is ``None`` for n<2."""
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return float("nan"), None
    if len(xs) == 1:
        return float(xs[0]), None
    return float(np.mean(xs)), float(np.std(xs, ddof=1))


def by_buffer(rows) -> pd.DataFrame:
    """Aggregate completed rows into a per-buffer table (mean +/- SD across seeds)."""
    out = []
    for buf in sorted({r["buffer_size"] for r in rows}):
        grp = [r for r in rows if r["buffer_size"] == buf]
        m, s = mean_std([r["old_fpr_fixed_mean"] for r in grp])
        oa, oas = mean_std([r["old_current_i_auroc_mean"] for r in grp])
        ca, cas = mean_std([r["old_acquisition_i_auroc_mean"] for r in grp])
        cu, _ = mean_std([r["current_task_i_auroc"] for r in grp])
        dr, drs = mean_std([r["old_normalized_score_drift_mean"] for r in grp])
        out.append({
            "buffer_size": int(buf),
            "n_runs": len(grp),
            "old_fpr_fixed_mean": m, "old_fpr_fixed_std": s,
            "old_current_i_auroc_mean": oa, "old_current_i_auroc_std": oas,
            "old_acquisition_i_auroc_mean": ca, "old_acquisition_i_auroc_std": cas,
            "current_task_i_auroc_mean": cu,
            "old_normalized_score_drift_mean": dr, "old_normalized_score_drift_std": drs,
        })
    return pd.DataFrame(out).sort_values("buffer_size").reset_index(drop=True)


def paper_table(by_buf: pd.DataFrame) -> pd.DataFrame:
    """Format the by-buffer table the way the paper reports it (percent +- SD)."""
    tbl = pd.DataFrame({
        "buffer": by_buf["buffer_size"],
        "old_fpr_fixed": by_buf["old_fpr_fixed_mean"],
        "old_current_i_auroc": by_buf["old_current_i_auroc_mean"],
        "old_acquisition_i_auroc": by_buf["old_acquisition_i_auroc_mean"],
        "current_task_i_auroc": by_buf["current_task_i_auroc_mean"],
        "drift": by_buf["old_normalized_score_drift_mean"],
    })
    for c in tbl.columns.difference(["buffer"]):
        tbl[c] = (tbl[c] * 100).round(1)
    return tbl
