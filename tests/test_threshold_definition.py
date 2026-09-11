"""Tests for the threshold definition (docs/threshold_definition.md).

These pin the semantics of ``tau_t``, ``sigma_t`` and the strict decision rule that the
paper's Eq.(1) depends on. They are hand-computable and need no run artifact.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import operating_point as op  # noqa: E402

N, A = op.NORMAL, op.ANOMALY


def _mk(nsc, asc):
    s = np.array(list(nsc) + list(asc), float)
    l = np.array([N] * len(nsc) + [A] * len(asc), int)
    return s, l


def test_tau_is_099_quantile_of_normals_not_all_scores():
    # Only the NORMAL scores feed tau; anomaly scores never influence it.
    s, l = _mk([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [999.0])
    tau, _ = op.tau_sigma(s, l, quantile=0.99)
    normals = np.sort(s[l == N])
    assert tau == pytest.approx(np.quantile(normals, 0.99))


def test_sigma_is_sample_sd_of_normals():
    s, l = _mk([0.0, 2.0, 4.0], [10.0])
    tau, sigma = op.tau_sigma(s, l, quantile=0.99)
    assert sigma == pytest.approx(np.std(s[l == N], ddof=1))


def test_strict_gt_means_equal_tau_is_fnr_not_fpr():
    # A normal score exactly == tau: strict > means NOT an alarm -> FPR = 0.
    s, l = _mk([0.5], [0.5])
    fpr, fnr, nn, na = op.metrics(s, l, tau=0.5)
    assert fpr == pytest.approx(0.0)
    assert fnr == pytest.approx(1.0)
    assert (nn, na) == (1, 1)


def test_threshold_at_delta_zero_equals_tau():
    assert op.threshold_at(1.2, 0.3, 0.0) == pytest.approx(1.2)


def test_threshold_at_delta_uses_sigma_units():
    assert op.threshold_at(1.0, 0.5, 2.0) == pytest.approx(2.0)


def test_standardized_score_is_z():
    s, _ = _mk([1.5], [0.0])
    z = op.standardized_scores(np.array([2.5]), 1.0, 0.5)
    assert z == pytest.approx(3.0)
