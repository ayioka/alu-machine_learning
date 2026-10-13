#!/usr/bin/env python3
"""Calculate the maximization step in the EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    # Ensure g forms valid posterior probabilities across clusters
    if not np.allclose(np.sum(g, axis=0), 1.0):
        return None, None, None

    # Effective number of points assigned to each cluster
    Nk = np.sum(g, axis=1)

    # Updated priors
    pi = Nk / n

    # Updated means
    m = np.matmul(g, X) / Nk[:, np.newaxis]

    # Updated covariance matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i]
        S[i] = np.matmul(g[i] * diff.T, diff) / Nk[i]

    return pi, m, S
