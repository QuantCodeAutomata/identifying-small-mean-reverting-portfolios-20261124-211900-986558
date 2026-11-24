"""Unit tests for the greedy sparse generalised eigenvalue search."""

from __future__ import annotations

import numpy as np

from src.box_tiao import canonical_decomposition, estimate_var1
from src.greedy import GreedyPath, greedy_search


def test_greedy_path_shapes(synthetic_var1):
    _, S_hat = estimate_var1(synthetic_var1)
    S_cur = synthetic_var1[1:]
    A = S_hat.T @ S_hat
    B = S_cur.T @ S_cur
    path = greedy_search(A, B, minimise=True)
    n = A.shape[0]
    assert len(path.weights) == n
    assert path.objective.shape == (n,)
    for k, w in enumerate(path.weights, start=1):
        assert w.shape == (n,)
        np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-8)
        assert np.sum(np.abs(w) > 1e-10) <= k


def test_greedy_full_support_matches_dense(synthetic_var1):
    res = canonical_decomposition(synthetic_var1)
    _, S_hat = estimate_var1(synthetic_var1)
    S_cur = synthetic_var1[1:]
    A = S_hat.T @ S_hat
    B = S_cur.T @ S_cur
    path = greedy_search(A, B, minimise=True)
    # At full cardinality the greedy objective must match the dense solution.
    np.testing.assert_allclose(path.objective[-1], res.predictability[0], rtol=1e-7)


def test_greedy_supports_nested(synthetic_var1):
    _, S_hat = estimate_var1(synthetic_var1)
    S_cur = synthetic_var1[1:]
    A = S_hat.T @ S_hat
    B = S_cur.T @ S_cur
    path = greedy_search(A, B, minimise=True)
    for k in range(1, len(path.supports)):
        assert set(path.supports[k - 1]).issubset(set(path.supports[k]))


def test_greedy_minimise_vs_maximise(synthetic_var1):
    _, S_hat = estimate_var1(synthetic_var1)
    S_cur = synthetic_var1[1:]
    A = S_hat.T @ S_hat
    B = S_cur.T @ S_cur
    lo = greedy_search(A, B, minimise=True).objective
    hi = greedy_search(A, B, minimise=False).objective
    assert np.all(hi >= lo - 1e-8)
