"""Tests for FPR/FNR semantics across the delta sweep (Fig.3 panels c/d).

These pin that FPR is non-increasing and FNR non-decreasing in ``delta`` (the whole point of
the sensitivity story) and that ``delta = 0`` equals the stored operating point.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import operating_point as op  # noqa: E402

N, A = op.NORMAL, op.ANOMALY


def _mk(nsc, asc):
    s = np.array(list(nsc) + list(asc), float)
    l = np.array([N] * len(nsc) + [A] * len(asc), int)
    return s, l


def test_fpr_non_increasing_fnr_non_decreasing_in_delta():
    s, l = _mk([1.0, 2.0, 3.0], [4.0, 5.0])
    tau, sigma = 1.5, 1.0
    fprs, fnrs = [], []
    for d in np.arange(-2.0, 4.0 + 1e-9, 0.25):
        thr = op.threshold_at(tau, sigma, d)
        fpr, fnr, _, _ = op.metrics(s, l, thr)
        fprs.append(fpr)
        fnrs.append(fnr)
    fprs = np.array(fprs)
    fnrs = np.array(fnrs)
    assert np.all(np.diff(fprs) <= 1e-12 + 0)  # FPR never increases
    assert np.all(np.diff(fnrs) >= -1e-12)      # FNR never decreases


def test_delta_zero_is_stored_operating_point():
    s, l = _mk([1.0, 2.0, 3.0], [4.0, 5.0])
    tau, sigma = 2.0, 0.7
    fpr0, fnr0, _, _ = op.metrics(s, l, tau)
    fpr_d, fnr_d, _, _ = op.metrics(s, l, op.threshold_at(tau, sigma, 0.0))
    assert fpr_d == pytest.approx(fpr0)
    assert fnr_d == pytest.approx(fnr0)


def test_monotone_crossing_at_breakpoint():
    # A single normal at 2.0 and a single anomaly at 4.0, tau=0, sigma=1:
    #   delta >= 2.0 => FPR=0;  delta < 4.0 => FNR=0. So the FPR/FNR cross between them.
    s, l = _mk([2.0], [4.0])
    tau, sigma = 0.0, 1.0
    for d in (-1.0, 1.0, 2.0, 3.0, 5.0):
        fpr, fnr, _, _ = op.metrics(s, l, op.threshold_at(tau, sigma, d))
        assert fpr in (0.0, 1.0)
        assert fnr in (0.0, 1.0)
    # d=2.0 exactly: normal score == tau+2*1 => strict > => FPR 0
    fpr, _, _, _ = op.metrics(s, l, op.threshold_at(tau, sigma, 2.0))
    assert fpr == pytest.approx(0.0)
    # d just below 4.0: anomaly score > threshold => FNR 0 (strict >, anomaly not <= thr)
    fpr, fnr, _, _ = op.metrics(s, l, op.threshold_at(tau, sigma, 3.99))
    assert fnr == pytest.approx(0.0)


def test_feasible_region_consistency():
    # The feasible region for FPR<=a AND FNR<=b must sit strictly between the FPR breakpoint
    # and the FNR breakpoint. Here normals [1,2,3], anomaly [4,5], tau=0, sigma=1.
    s, l = _mk([1.0, 2.0, 3.0], [4.0, 5.0])
    iv, _, _ = op.feasible_delta(s, l, tau=0.0, sigma=1.0, alpha=1 / 3, beta=0.5)
    if iv:
        lo, hi = iv[0]
        for d in np.linspace(lo + 1e-6, hi - 1e-6, 20):
            fpr, fnr, _, _ = op.metrics(s, l, op.threshold_at(0.0, 1.0, d))
            assert fpr <= 1 / 3 + 1e-9
            assert fnr <= 0.5 + 1e-9
