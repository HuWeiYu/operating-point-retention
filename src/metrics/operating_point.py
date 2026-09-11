#!/usr/bin/env python3
"""Operating-point / threshold-semantics utilities for the (UCAD) audit.

This module implements the paper's Eq.(1) audit protocol on per-image endpoint
scores. It is **pure NumPy** (no GPU, no absolute paths, no training code) so the
statistics can be recomputed offline from already-published score artifacts.

Definitions encoded here (these are the audited semantics, not a summary):

* **Per-task stored threshold** ``tau_t`` = the 0.99 quantile of the acquisition
  *normal-validation* scores of task ``t``. It is stored at acquisition time and
  **not** re-tuned: there is **no** test-set recalibration.

* **Decision rule** (Eq. 1): alarm iff ``score > tau`` (strict). A score **equal**
  to ``tau`` is *not* an alarm, so it counts toward FNR but not toward FPR.

* **FPR** = Pr[ normal score > tau ]  (empirical: fraction of normal test images
  alarmed).  **FNR** = Pr[ anomaly score <= tau ] (fraction of anomaly test images
  missed).  Both use the **same** stored ``tau``.

* **AUROC** is image-level (see :func:`image_auroc`); pixel-level metrics, if any,
  are diagnostics only.

* **Post-hoc sensitivity sweep.** Moving the threshold to ``tau + delta*sigma_t``
  changes the alarm rule to ``score > tau + delta*sigma_t``. Writing the
  standardized score ``z = (score - tau)/sigma_t``, the rule becomes ``z > delta``.
  This makes ``delta`` dimensionless and - critically - a **common ``delta`` is not
  a common absolute threshold shift** ``Delta_tau``, because each task has its own
  ``sigma_t``.

* **``delta = 0``** must reproduce the stored operating point and therefore the
  formal ``FPR@fixed`` / ``FNR@fixed`` artifacts (a property the audit verifies).

Inputs are plain score/label arrays. Callers read them from their own artifacts
(e.g. an ``.npz`` bundle with ``scores`` / ``labels`` / ``paths`` arrays) and pass
arrays here; this package never reads absolute paths itself.

"Example tolerance" below (FPR<=5% AND FNR<=10%) is an **example**, not an
industrial standard, and any threshold chosen on the test set is a statement about
the measured error rates, **not** a claim of deployment calibration benefit.
"""
from __future__ import annotations

import numpy as np

#: label numeric encoding used by the run artifacts
NORMAL, ANOMALY = 0, 1

#: example condition used for the feasible-region reports (an example only)
EX_FPR, EX_FNR = 0.05, 0.10


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
def tau_sigma(scores: np.ndarray, labels: np.ndarray, quantile: float = 0.99) -> tuple:
    """Per-task stored ``tau`` and ``sigma`` from a task's acquisition scores.

    ``tau`` is the ``quantile`` of the **normal-validation** scores (labels==NORMAL);
    ``sigma`` is their standard deviation. ``quantile`` default 0.99 per the paper.

    Returns ``(tau, sigma)`` as float (``sigma`` is ``nan`` if fewer than 2 normals).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    nsc = scores[labels == NORMAL]
    if nsc.size == 0:
        return float("nan"), float("nan")
    tau = float(np.quantile(nsc, quantile))
    sigma = float(nsc.std(ddof=1)) if nsc.size > 1 else float("nan")
    return tau, sigma


def threshold_at(tau: float, sigma: float, delta: float) -> float:
    """Threshold ``tau + delta*sigma`` used by the post-hoc sweep."""
    return tau + delta * sigma


# --------------------------------------------------------------------------- #
# decision metrics at a single threshold
# --------------------------------------------------------------------------- #
def metrics(scores: np.ndarray, labels: np.ndarray, tau: float):
    """Empirical (FPR, FNR, n_normal, n_anomaly) under strict ``score > tau``.

    * FPR = mean( normal scores > tau )
    * FNR = mean( anomaly scores <= tau )

    ``nan`` for a term whose class is absent. Returned as a 4-tuple of float/int.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n = labels == NORMAL
    a = labels == ANOMALY
    fpr = float(np.mean(scores[n] > tau)) if n.any() else float("nan")
    fnr = float(np.mean(scores[a] <= tau)) if a.any() else float("nan")
    return fpr, fnr, int(n.sum()), int(a.sum())


def decision(scores: np.ndarray, labels: np.ndarray, tau: float) -> np.ndarray:
    """Boolean alarm decision ``score > tau`` (strict); same convention as :func:`metrics`."""
    return np.asarray(scores, dtype=float) > tau


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level AUROC from per-image anomaly scores (positive class = anomaly).

    Scores may be image-level already; if a detector returns per-pixel maps, the
    caller must reduce them to one score *per image* before calling (pixel-level
    AUROC is a diagnostic and is not what Eq.(1)/Table I report).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels) == ANOMALY
    if labels.all() or not labels.any():
        return float("nan")
    # rank-based AUC, avoids sklearn dependency; ties averaged
    order = scores.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    # tie handling: give tied points the mean rank
    s_sorted = scores[order]
    _, idx, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for i, c in zip(idx, counts):
        ranks[order[i : i + c]] = ranks[order[i : i + c]].mean()
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    pos_ranks = ranks[labels].sum()
    return float((pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# exact empirical feasible region of delta (audited, no rounding)
# --------------------------------------------------------------------------- #
def standardized_scores(scores: np.ndarray, tau: float, sigma: float) -> np.ndarray:
    """z = (score - tau)/sigma. Guarded against sigma<=0 (returns empty)."""
    if sigma is None or sigma <= 0 or np.isnan(sigma):
        return np.array([], dtype=float)
    return (np.asarray(scores, dtype=float) - tau) / sigma


def feasible_delta(scores: np.ndarray, labels: np.ndarray, tau: float, sigma: float,
                   alpha: float = EX_FPR, beta: float = EX_FNR):
    """Exact set of ``delta`` with FPR<=alpha AND FNR<=beta (half-open, unrounded).

    The decision rule ``z > delta`` makes FPR and FNR step functions of ``delta``
    whose only breakpoints are the DISTINCT standardized scores. So on each half-open
    interval ``[b_i, b_{i+1})``:

        FPR = #{normal z >= b_{i+1}} / N_n   (strict > excludes scores == b_i)
        FNR = #{anomaly z <= b_i}     / N_a  (scores == b_i are included at delta==b_i)

    The unbounded edges are ``(-inf, b_0) -> (FPR=1, FNR=0)`` and ``[b_{m-1}, +inf)
    -> (FPR=0, FNR=1)``. Duplicate equal normal/anomaly breakpoints collapse via
    ``np.unique``. FPR is non-increasing and FNR non-decreasing in ``delta``, so the
    satisfying set is one contiguous half-open segment (possibly empty, possibly
    unbounded on one side), returned merged.

    Returns ``(intervals, nsc, asc)`` where ``intervals`` is a list of ``(lo, hi)``
    tuples (floats or +/-inf) and ``nsc``/``asc`` are the standardized normal/anomaly
    arrays (handy for plotting).
    """
    n = labels == NORMAL
    a = labels == ANOMALY
    if sigma is None or sigma <= 0 or np.isnan(sigma):
        return [], [], []
    nsc = (np.asarray(scores[n], dtype=float) - tau) / sigma
    asc = (np.asarray(scores[a], dtype=float) - tau) / sigma
    bp = np.unique(np.concatenate([nsc, asc]))  # distinct breakpoints, ascending
    cnt_n, cnt_a = nsc.size, asc.size

    segs = []
    left = -np.inf
    for j, right in enumerate(bp):
        right = float(right)
        fpr = float(np.count_nonzero(nsc >= right)) / cnt_n
        fnr = float(np.count_nonzero(asc <= left)) / cnt_a if j > 0 else 0.0
        segs.append((left, right, fpr, fnr))
        left = right
    fnr = float(np.count_nonzero(asc <= left)) / cnt_a
    segs.append((left, np.inf, 0.0, fnr))

    satisfying = [(float(s0), float(s1)) for s0, s1, fpr, fnr in segs
                  if fpr <= alpha and fnr <= beta]
    merged = []
    for lo, hi in satisfying:
        if merged and lo == merged[-1][1]:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return merged, nsc, asc


def fmt_interval(iv) -> str:
    """Human-readable ``'[lo, hi)'`` string, inf-safe, 6-decimal (never rounded)."""
    if not iv:
        return "EMPTY"
    lo, hi = iv

    def f(x):
        if np.isneginf(x):
            return "-inf"
        if np.isposinf(x):
            return "+inf"
        return f"{x:.6f}"

    return f"[{f(lo)}, {f(hi)})"


def interval_width(iv):
    if not iv:
        return 0.0
    lo, hi = iv
    if np.isneginf(lo) or np.isposinf(hi):
        return np.inf
    return hi - lo


def parse_interval_str(s):
    """Parse ``'[lo, hi); ...'`` (or ``'EMPTY'``) back to a list of ``(lo,hi)``."""
    if not s or str(s) == "EMPTY":
        return []
    out = []
    for seg in str(s).split(";"):
        seg = seg.strip()
        if not seg:
            continue
        inner = seg.strip("[] ").rstrip(")")
        lo, hi = (x.strip() for x in inner.split(","))
        lo = -np.inf if lo == "-inf" else float(lo)
        hi = np.inf if hi == "+inf" else float(hi)
        out.append((lo, hi))
    return out


def intersect_intervals(inters, others):
    """Intersection of two list-of-(lo,hi). Empty on no overlap."""
    out = []
    for la, lb in inters:
        for ra, rb in others:
            lo, hi = max(la, ra), min(lb, rb)
            if lo <= hi:
                out.append((lo, hi))
    return out


def common_feasible_delta(per_task_intervals, old_tasks=(1, 2, 3, 4)):
    """Intersect the feasible-delta sets over a set of old tasks."""
    common = None
    for t in old_tasks:
        iv = per_task_intervals.get(int(t), [])
        common = list(iv) if common is None else intersect_intervals(common, iv)
    return common or []


# --------------------------------------------------------------------------- #
# aggregation helpers (the paper's mean/SD is over seeds, of per-task means)
# --------------------------------------------------------------------------- #
def mean_std(xs):
    """Mean and sample SD (ddof=1) over a list, ignoring nan entries."""
    xs = np.asarray([x for x in xs if not (isinstance(x, float) and np.isnan(x))],
                    dtype=float)
    if xs.size == 0:
        return float("nan"), float("nan")
    if xs.size == 1:
        return float(xs[0]), float("nan")
    return float(xs.mean()), float(xs.std(ddof=1))
