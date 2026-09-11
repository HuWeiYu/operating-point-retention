"""Read-only loaders for saved run artifacts.

A "run" on disk is a directory with, minimally, ``config.json``, ``stage_metrics.csv``,
``drift_metrics.csv`` and per-image ``.npz`` score bundles under ``stage_SS/``. This
module reads those. It performs **no** computation beyond parsing and never writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def read_config(run_dir) -> dict:
    """Return the run's ``config.json`` as a dict, or ``{}`` if absent."""
    p = Path(run_dir) / "config.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_bundle(npz_path) -> dict:
    """Load an ``.npz`` per-image score bundle.

    Returns ``{"paths", "categories", "labels", "scores", "routes"}``; ``routes`` is
    zeros when the bundle has no routing column (a shared detector).
    """
    d = np.load(str(npz_path), allow_pickle=True)
    return {
        "paths": d["paths"].astype(str),
        "categories": d["categories"].astype(str),
        "labels": d["labels"].astype(np.int32),
        "scores": d["scores"].astype(np.float32),
        "routes": (d["routes"].astype(np.int32)
                   if "routes" in d else np.zeros(len(d["labels"]), np.int32)),
    }


def _bundle_path(run_dir, stage, task, suffix):
    """Path to ``stage_SS/task_tt<suffix>.npz`` (validation/test variants)."""
    return Path(run_dir) / f"stage_{stage:02d}" / f"task_{task:02d}{suffix}.npz"


def stored_tau(run_dir, task):
    """The stored per-task acquisition threshold ``tau_t`` from ``stage_SS/train.json``.

    Returns a float, or ``None`` if the run has no ``train.json`` or no ``threshold``
    field. This is the value Eq.(1) uses: it is **stored at acquisition and never
    recalibrated** from test scores.
    """
    p = Path(run_dir) / f"stage_{task:02d}" / "train.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    t = d.get("threshold")
    return float(t) if t is not None else None


def stored_validation_stats(run_dir, task):
    """``(sigma_t, q99)`` from the run's ``validation_metrics.csv`` for task ``t``.

    ``sigma_t`` (``validation_score_std``) and ``q99`` (``validation_score_q99``) are the
    acquisition normal-validation summary the sweep needs; ``q99`` cross-checks the stored
    ``tau_t``. Returns ``(None, None)`` if the row or columns are missing.
    """
    import pandas as pd
    p = Path(run_dir) / "validation_metrics.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    row = df[(df["eval_task"] == task) & (df["stage"] == task)]
    if row.empty:
        return None, None
    last = row.iloc[-1]
    sigma = last.get("validation_score_std")
    q99 = last.get("validation_score_q99")
    return (float(sigma) if sigma is not None and not (isinstance(sigma, float) and np.isnan(sigma)) else None,
            float(q99) if q99 is not None else None)


def acquisition_validation_bundle(run_dir, task):
    """Acquisition normal-validation bundle for task ``t`` (``_validation_shared``, then oracle)."""
    for suffix in ("_validation_shared", "_validation_oracle"):
        p = _bundle_path(run_dir, task, task, suffix)
        if p.exists():
            return p
    return None


def endpoint_bundle(run_dir, tail_stage, task, route="shared"):
    """Endpoint (``tail_stage``) score bundle for old task ``task`` under ``route``."""
    return _bundle_path(run_dir, tail_stage, task, f"_{route}")


def load_stage_metrics(run_dir) -> "pd.DataFrame" | None:
    """Return ``stage_metrics.csv`` as a DataFrame, or ``None`` if absent."""
    import pandas as pd
    p = Path(run_dir) / "stage_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


def load_run_dir(run_dir) -> dict | None:
    """Bundle the artefacts of one run into a dict, or ``None`` if it is incomplete.

    Returns ``{"config", "stage", "drift", "buffer", "method", "seed"}``. A run is
    considered complete when both ``stage_metrics.csv`` and ``drift_metrics.csv`` exist.
    """
    import pandas as pd
    cfg = read_config(run_dir)
    stage_p = Path(run_dir) / "stage_metrics.csv"
    drift_p = Path(run_dir) / "drift_metrics.csv"
    if not stage_p.exists() or not drift_p.exists():
        return None
    stage = pd.read_csv(stage_p)
    drift = pd.read_csv(drift_p)
    for df in (stage, drift):
        for col in ("stage", "eval_task"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return {
        "dir": Path(run_dir),
        "config": cfg,
        "stage": stage,
        "drift": drift,
        "buffer": int(cfg.get("replay_buffer_size", 0)),
        "method": cfg.get("method", "shared_ft"),
        "seed": int(cfg.get("seed", 1)),
    }
