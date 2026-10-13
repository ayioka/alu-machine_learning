#!/usr/bin/env python3
"""Initialize Gaussian Mixture Model parameters."""
import numpy as np


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    n, d = X.shape

    # Evenly initialized priors: shape (k,)
    pi = np.full((k,), 1.0 / k)

    # Centroids initialized with multivariate uniform distribution: shape (k, d)
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    m = np.random.uniform(low, high, size=(k, d))

    # Covariance matrices initialized as identity matrices: shape (k, d, d)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
