"""NumPy-only tests for the operating-point metrics and the audited recomputation.

These pin the metric semantics that Eq.(1) and Fig.3 rely on, and the ``delta = 0``
reproduction of the stored operating point. Pure NumPy — no GPU, no data, no model.

Run:  python -m pytest tests/test_metrics_repro.py  (or pytest if installed)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import operating_point as op  # noqa: E402

N, A = op.NORMAL, op.ANOMALY


def _mk(nsc, asc):
    s = np.array(list(nsc) + list(asc), dtype=float)
    l = np.array([N] * len(nsc) + [A] * len(asc), dtype=int)
    return s, l


# --- tau_sigma: stored per-task threshold = quantile of acquisition normals --------- #
def test_tau_is_quantile_of_normals():
    normals = np.array([0.0] * 200 + [5.0])  # >99% zeros -> 0.99-quantile is 0.0
    labels = np.array([N] * normals.size)
    tau, sigma = op.tau_sigma(normals, labels, quantile=0.99)
    assert tau == pytest.approx(float(np.quantile(normals, 0.99)))
    assert sigma == pytest.approx(float(normals.std(ddof=1)))
    # 0.99-quantile of a set that is >99% zeros is exactly 0.0
    assert tau == pytest.approx(0.0)


def test_tau_sigma_uses_only_normals():
    # anomalies must NOT influence tau/sigma
    normals = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    anoms = np.array([5.0, 6.0, 7.0])
    s = np.concatenate([normals, anoms]); l = np.array([N] * 5 + [A] * 3)
    tau, sigma = op.tau_sigma(s, l)
    assert tau == pytest.approx(float(np.quantile(normals, 0.99)))
    assert sigma == pytest.approx(float(normals.std(ddof=1)))


# --- metrics: strict score > tau ------------------------------------------------ #
def test_metrics_strict_greater_and_fnr_pair():
    s, l = _mk([1, 2, 3, 3.5], [4, 5, 6, 7])  # one normal exactly AT tau
    tau = 3.5
    fpr, fnr, nn, na = op.metrics(s, l, tau)
    assert fpr == pytest.approx(0.0)      # normal == tau is NOT flagged (strict, > only)
    assert fnr == pytest.approx(0.0)      # all anomalies > tau -> none <= tau
    assert (nn, na) == (4, 4)


def test_metrics_fpr_matches_mean_decision():
    s, l = _mk([1, 2, 3, 10], [4, 5, 6, 7])
    tau = 2.0
    fpr, fnr, nn, na = op.metrics(s, l, tau)
    assert fpr == pytest.approx(np.mean(np.asarray(s)[l == N] > 2.0))
    assert fnr == pytest.approx(np.mean(np.asarray(s)[l == A] <= 2.0))
    assert fpr == pytest.approx(0.5)   # normals: 1,2,3,10 -> only 1,2 <= 2


# --- threshold calibration: delta=0 == stored operating point --------------------- #
def test_delta0_reproduces_stored_operating_point():
    rng = np.random.default_rng(2026)
    normals = rng.normal(0.0, 1.0, 5000)
    anoms = rng.normal(3.0, 1.0, 500)
    s = np.concatenate([normals, anoms]); l = np.array([N] * 5000 + [A] * 500)
    tau, sigma = op.tau_sigma(s, l, quantile=0.99)
    # stored operating point is at delta=0 -> threshold = tau
    thr = op.threshold_at(tau, sigma, 0.0)
    assert thr == pytest.approx(tau)
    fpr, fnr, _, _ = op.metrics(s, l, thr)
    # with 5000 normals, the 0.99-quantile threshold makes FPR ~ 1%
    assert fpr == pytest.approx(0.01, abs=0.01)
    assert 0.0 <= fpr <= 0.03
    assert 0.0 <= fnr <= 1.0


def test_delta_monotonicity_fpr_fnr():
    rng = np.random.default_rng(7)
    s = np.concatenate([rng.normal(0, 1, 2000), rng.normal(2.5, 1, 200)])
    l = np.array([N] * 2000 + [A] * 200)
    tau, sigma = op.tau_sigma(s, l)
    prev_fpr, prev_fnr = 2.0, -1.0
    for d in np.arange(-3.0, 4.0, 0.5):
        fpr, fnr, _, _ = op.metrics(s, l, op.threshold_at(tau, sigma, float(d)))
        assert fpr <= prev_fpr + 1e-9   # FPR non-increasing in delta
        assert fnr >= prev_fnr - 1e-9   # FNR non-decreasing in delta
        prev_fpr, prev_fnr = fpr, fnr


# --- image_auroc: rank-based, ties averaged --------------------------------------- #
def test_image_auroc_perfect_and_worst():
    s, l = _mk([0, 1, 2, 3], [10, 11, 12, 13])
    assert op.image_auroc(s, l) == pytest.approx(1.0)
    # single-class input -> AUROC undefined -> nan (0/0), not 1.0
    assert np.isnan(op.image_auroc(np.array([10, 11, 12, 13]), np.array([0, 0, 0, 0])))
    assert np.isnan(op.image_auroc(np.array([10, 11, 12, 13]), np.array([1, 1, 1, 1])))


def test_image_auroc_reversed_and_ties():
    s, l = _mk([10, 11, 12, 13], [0, 1, 2, 3])   # anomalies score LOWER -> 0.0
    assert op.image_auroc(s, l) == pytest.approx(0.0)
    # all ties -> ranks averaged -> 0.5
    s_tie, l_tie = _mk([5, 5, 5], [5, 5, 5])
    assert op.image_auroc(s_tie, l_tie) == pytest.approx(0.5)
