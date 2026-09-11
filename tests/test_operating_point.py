#!/usr/bin/env python3
"""Unit tests for :mod:`src.metrics.operating_point`.

These are **hand-computable** checks that do not need any run artifact, so the metric
module can be trusted before it is pointed at real data. They encode the exact
half-open-breakpoint behaviour of the audited FPR/FNR definition.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import operating_point as op  # noqa: E402

N, A = op.NORMAL, op.ANOMALY


def _scores_normal_anomaly(nsc, asc):
    scores = np.array(list(nsc) + list(asc), dtype=float)
    labels = np.array([N] * len(nsc) + [A] * len(asc), dtype=int)
    return scores, labels


def test_metrics_strict_and_half_open():
    # two normals at 0.1/0.9, two anomalies at 0.5/0.9; tau=0.5
    # strict score>tau: normal 0.9 -> FPR=1/2; anomaly <=0.5 -> 0.5 only -> FNR=1/2
    scores, labels = _scores_normal_anomaly([0.1, 0.9], [0.5, 0.9])
    fpr, fnr, nn, na = op.metrics(scores, labels, tau=0.5)
    assert fpr == pytest.approx(0.5)
    assert fnr == pytest.approx(0.5)
    assert (nn, na) == (2, 2)
    # a normal score exactly == tau is NOT alarmed (strict >), so FPR=0 for it;
    # an anomaly exactly == tau IS missed (<=), so FNR=1.
    scores, labels = _scores_normal_anomaly([0.5], [0.5])
    fpr, fnr, _, _ = op.metrics(scores, labels, tau=0.5)
    assert fpr == pytest.approx(0.0)
    assert fnr == pytest.approx(1.0)


def test_tau_is_099_quantile_of_normals():
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    labels = np.array([N] * 10)
    tau, sigma = op.tau_sigma(scores, labels, quantile=0.99)
    assert tau == pytest.approx(np.quantile(scores, 0.99))
    assert sigma == pytest.approx(scores.std(ddof=1))


def test_decision_matches_metrics():
    scores, labels = _scores_normal_anomaly([0.1, 0.9], [0.5, 0.9])
    dec = op.decision(scores, labels, tau=0.5)
    fpr, fnr, _, _ = op.metrics(scores, labels, tau=0.5)
    n = labels == N
    a = labels == A
    assert dec[n].mean() == pytest.approx(fpr)
    assert (reset := np.mean(~dec[a])) == pytest.approx(fnr)


def test_image_auroc_rank_method():
    # perfect separation: low normals, high anomalies -> AUC=1
    scores, labels = _scores_normal_anomaly([0.1, 0.2, 0.3], [4.0, 5.0, 6.0])
    assert op.image_auroc(scores, labels) == pytest.approx(1.0)
    # reversed -> AUC=0
    scores, labels = _scores_normal_anomaly([4.0, 5.0, 6.0], [0.1, 0.2, 0.3])
    assert op.image_auroc(scores, labels) == pytest.approx(0.0)
    # ties handled without error (rank-average path)
    scores, labels = _scores_normal_anomaly([1.0, 1.0], [1.0, 2.0])
    assert 0.0 <= op.image_auroc(scores, labels) <= 1.0


def test_feasible_nonzero_width():
    # normals at 0.5, anomaly at 2.0, alpha=beta=0.5 -> delta in [0.5, 2.0)
    scores, labels = _scores_normal_anomaly([0.5], [2.0])
    iv, nsc, asc = op.feasible_delta(scores, labels, tau=0.0, sigma=1.0,
                                     alpha=0.5, beta=0.5)
    assert op.fmt_interval(iv[0]) == "[0.500000, 2.000000)"
    assert op.interval_width((0.5, 2.0)) == 1.5


def test_feasible_empty():
    # normals at 10, anomaly at 0 -> FPR<=.5 needs delta>=10, FNR<=.5 needs delta<0
    scores, labels = _scores_normal_anomaly([10.0], [0.0])
    iv, _, _ = op.feasible_delta(scores, labels, tau=0.0, sigma=1.0, alpha=0.5, beta=0.5)
    assert iv == []


def test_feasible_duplicates_collapse():
    # two normals + one anomaly all == 1.0, alpha=0.5 beta=0.6 -> EMPTY
    scores, labels = _scores_normal_anomaly([1.0, 1.0], [1.0])
    iv, _, _ = op.feasible_delta(scores, labels, tau=0.0, sigma=1.0, alpha=0.5, beta=0.6)
    assert iv == []


def test_feasible_equality_half_open():
    # normals 1.0, anomaly 2.0, alpha=beta=0.5 -> [1.0, 2.0)
    scores, labels = _scores_normal_anomaly([1.0], [2.0])
    iv, _, _ = op.feasible_delta(scores, labels, tau=0.0, sigma=1.0, alpha=0.5, beta=0.5)
    assert op.fmt_interval(iv[0]) == "[1.000000, 2.000000)"


def test_feasible_merge_full_line():
    # nsc=[3.0], asc=[1.0], alpha=beta=1.0 -> whole line
    scores, labels = _scores_normal_anomaly([3.0], [1.0])
    iv, _, _ = op.feasible_delta(scores, labels, tau=0.0, sigma=1.0, alpha=1.0, beta=1.0)
    assert op.fmt_interval(iv[0]) == "[-inf, +inf)"


def test_interval_roundtrip():
    s = "[0.500000, 2.000000);[-inf, +inf)"
    ivs = op.parse_interval_str(s)
    assert ivs == [(0.5, 2.0), (-np.inf, np.inf)]
    assert op.fmt_interval(ivs[0]) == "[0.500000, 2.000000)"
    assert op.fmt_interval(ivs[1]) == "[-inf, +inf)"


def test_common_delta_intersection():
    per_task = {1: [(0.5, 2.0)], 2: [(1.0, 3.0)], 3: [(1.5, 2.5)], 4: []}
    assert op.common_feasible_delta(per_task, old_tasks=(1, 2, 3)) == [(1.5, 2.0)]
    assert op.common_feasible_delta(per_task, old_tasks=(1, 2, 3, 4)) == []


def test_mean_std():
    m, s = op.mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert s == pytest.approx(1.0, abs=1e-9)
    m, s = op.mean_std([float("nan"), 5.0])
    assert m == pytest.approx(5.0)
    assert np.isnan(s)
