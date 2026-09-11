"""Artifact loaders and aggregation for the operating-point audit.

This subpackage reads *saved* run artifacts (per-image ``.npz`` score bundles and
``stage_metrics.csv``) and aggregates them the way the paper does. It never launches
a training run. Paths are passed in by the caller; no absolute path is hardcoded.
"""
from .io import load_bundle, load_run_dir, read_config, load_stage_metrics, \
    stored_tau, stored_validation_stats, acquisition_validation_bundle, endpoint_bundle
from .aggregate import collect_runs, mean_std, by_buffer, paper_table

__all__ = [
    "load_bundle", "load_run_dir", "read_config", "load_stage_metrics",
    "stored_tau", "stored_validation_stats", "acquisition_validation_bundle", "endpoint_bundle",
    "collect_runs", "mean_std", "by_buffer", "paper_table",
]
