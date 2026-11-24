"""Unit tests for the Box & Tiao canonical decomposition."""

from __future__ import annotations

import numpy as np

from src.box_tiao import (
    BoxTiaoResult,
    canonical_decomposition,
    estimate_var1,
    predictability,
)


def test_estimate_var1_shape(synthetic_var1):
    A_hat, S_hat = estimate_var1(synthetic_var1)
    n = synthetic_var1.shape[1]
    assert A_hat.shape == (n, n)
    assert S_hat.shape == (synthetic_var1.shape[0] - 1, n)


def test_canonical_decomposition_eigenvalue_ordering(synthetic_var1):
    res = canonical_decomposition(synthetic_var1)
    # eigh returns ascending eigenvalues; mean reversion corresponds to the
    # smallest predictability, so the sequence must be non-decreasing.
    eigs = res.predictability
    assert np.all(np.diff(eigs) >= -1e-10)
    # Predictabilities are variance ratios, so they lie in [0, 1]
    assert np.all(eigs >= -1e-10)
    assert np.all(eigs <= 1 + 1e-8)


def test_weights_are_unit_norm(synthetic_var1):
    res = canonical_decomposition(synthetic_var1)
    norms = np.linalg.norm(res.weights, axis=0)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-8)


def test_predictability_helper_matches_eigenvalue(synthetic_var1):
    res = canonical_decomposition(synthetic_var1)
    # The dense eigenvalues in canonical_decomposition are the generalised
    # eigenvalues of (S_hat^T S_hat, S_t^T S_t). ``predictability`` re-expresses
    # them as x^T A^T Gamma A x / x^T Gamma x. For a VAR(1), S_hat = S_lag A,
    # so the matching Gamma plug-in is S_lag^T S_lag, not S_t^T S_t.
    S_lag = synthetic_var1[:-1]
    gamma = S_lag.T @ S_lag
    S_cur = synthetic_var1[1:]
    gamma_cur = S_cur.T @ S_cur
    for k in range(res.weights.shape[1]):
        x = res.weights[:, k]
        num = x @ res.A_hat.T @ gamma @ res.A_hat @ x
        den = x @ gamma_cur @ x
        np.testing.assert_allclose(num / den, res.predictability[k],
                                   rtol=1e-8, atol=1e-10)
