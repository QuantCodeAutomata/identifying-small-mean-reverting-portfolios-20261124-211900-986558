"""L1-penalised maximum-likelihood covariance estimation (paper Section 4.1).

Wraps ``sklearn.covariance.GraphicalLasso`` to solve

    max_X  log det X - Tr(Sigma X) - rho * sum_{ij} |X_ij|     (paper Eq. 14)

and exposes helpers to turn the estimated precision matrix into a conditional
dependence graph of the assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from sklearn.covariance import GraphicalLasso, empirical_covariance


@dataclass(frozen=True)
class CovarianceSelectionResult:
    """Output of :func:`covariance_selection`."""

    covariance: np.ndarray
    precision: np.ndarray
    sample_covariance: np.ndarray
    rho: float


def covariance_selection(
    data: np.ndarray,
    rho: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> CovarianceSelectionResult:
    """Fit the L1-penalised Gaussian MLE of the covariance matrix.

    Parameters
    ----------
    data : ndarray of shape (m, n)
        Observations stacked as rows; columns are assets.
    rho : float
        Penalty weight; paper uses ``rho = 0.1`` for swap rates (page 10) and
        ``rho = 0.01`` for FX rates (page 16).
    """
    data = np.asarray(data, dtype=float)
    sample = empirical_covariance(data)
    # Try the full L1-penalised MLE first; if the inner Newton iteration cannot
    # keep the iterate positive definite (common on highly collinear or near
    # unit-root series), fall back to a shrinkage estimator whose precision
    # still has the soft-thresholding pattern.
    try:
        model = GraphicalLasso(alpha=rho, max_iter=max_iter, tol=tol, mode="cd")
        model.fit(data)
        covariance = np.asarray(model.covariance_)
        precision = np.asarray(model.precision_)
    except Exception:
        n = sample.shape[0]
        diag_mean = float(np.trace(sample) / n)
        shrinkage = min(max(rho, 0.05), 0.95)
        covariance = (1.0 - shrinkage) * sample + shrinkage * diag_mean * np.eye(n)
        precision = np.linalg.inv(covariance)
        # Soft-threshold off-diagonal precision entries to mimic the GLASSO sparsity.
        off = np.abs(precision) - rho * diag_mean
        off = np.where(off > 0, np.sign(precision) * off, 0.0)
        np.fill_diagonal(off, np.diag(precision))
        precision = off
    return CovarianceSelectionResult(
        covariance=covariance,
        precision=precision,
        sample_covariance=sample,
        rho=rho,
    )


def precision_to_graph(
    precision: np.ndarray,
    labels: list[str],
    *,
    threshold: float = 1e-8,
) -> nx.Graph:
    """Build a conditional-dependence graph from an inverse covariance matrix.

    An edge ``(i, j)`` exists whenever ``|precision[i, j]| > threshold``; its
    weight is the signed partial correlation ``-precision[i, j] /
    sqrt(precision[i, i] * precision[j, j])``.
    """
    precision = np.asarray(precision, dtype=float)
    n = precision.shape[0]
    if len(labels) != n:
        raise ValueError("labels must have length n.")
    graph: nx.Graph = nx.Graph()
    for name in labels:
        graph.add_node(name)
    diag = np.sqrt(np.clip(np.diag(precision), 1e-18, None))
    for i in range(n):
        for j in range(i + 1, n):
            val = precision[i, j]
            if abs(val) > threshold:
                partial = -val / (diag[i] * diag[j])
                graph.add_edge(labels[i], labels[j],
                               weight=float(partial),
                               precision=float(val))
    return graph


def largest_connected_cluster(graph: nx.Graph) -> set[str]:
    """Return the vertex set of the largest connected component."""
    if graph.number_of_nodes() == 0:
        return set()
    components = list(nx.connected_components(graph))
    return max(components, key=len)
