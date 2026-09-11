"""Public, dependency-light metric primitives for the operating-point audit.

Everything in this subpackage is pure NumPy: it never reads an absolute path, never
touches a GPU, and operates only on ``(scores, labels)`` arrays the caller supplies.
"""
from .operating_point import (
    NORMAL,
    ANOMALY,
    EX_FPR,
    EX_FNR,
    tau_sigma,
    threshold_at,
    metrics,
    decision,
    image_auroc,
    standardized_scores,
    feasible_delta,
    fmt_interval,
    interval_width,
    parse_interval_str,
    intersect_intervals,
    common_feasible_delta,
    mean_std,
)

__all__ = [
    "NORMAL", "ANOMALY", "EX_FPR", "EX_FNR",
    "tau_sigma", "threshold_at", "metrics", "decision", "image_auroc",
    "standardized_scores", "feasible_delta", "fmt_interval", "interval_width",
    "parse_interval_str", "intersect_intervals", "common_feasible_delta", "mean_std",
]
