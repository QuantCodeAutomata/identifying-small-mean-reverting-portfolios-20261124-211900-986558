"""Pytest fixtures and shared configuration."""

from __future__ import annotations

import os

import numpy as np
import pytest


def pytest_configure(config):  # noqa: D401
    config.addinivalue_line(
        "markers",
        "market_data: test depends on a live market-data download.",
    )


@pytest.fixture(scope="session")
def synthetic_var1() -> np.ndarray:
    """Draw a small stationary VAR(1) series with a known structure."""
    rng = np.random.default_rng(0)
    n, m = 5, 300
    A = 0.4 * np.eye(n) + 0.1 * rng.standard_normal((n, n))
    A /= 1.5 * np.linalg.norm(A, ord=2)
    chol = np.tril(rng.standard_normal((n, n))) / np.sqrt(n)
    S = np.zeros((m + 50, n))
    for t in range(1, m + 50):
        S[t] = S[t - 1] @ A + rng.standard_normal(n) @ chol.T
    return S[50:]


@pytest.fixture(scope="session")
def ou_series() -> np.ndarray:
    """Synthetic OU series with a known mean-reversion speed."""
    rng = np.random.default_rng(7)
    mu, lam, sigma, dt = 1.0, 5.0, 0.1, 1.0 / 252.0
    n = 2000
    x = np.empty(n)
    x[0] = mu
    phi = np.exp(-lam * dt)
    noise_sd = sigma * np.sqrt((1 - np.exp(-2 * lam * dt)) / (2 * lam))
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + noise_sd * rng.standard_normal()
    return x


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_MARKET_DATA_TESTS"):
        return
    skip = pytest.mark.skip(reason="market_data tests disabled by default")
    for item in items:
        if "market_data" in item.keywords:
            item.add_marker(skip)
