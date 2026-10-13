#!/usr/bin/env python3
"""Test for the optimum number of clusters by variance."""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    n = X.shape[0]
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmin > kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    d_vars = []
    base_variance = None

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None

        if k == kmin:
            base_variance = var

        d_vars.append(base_variance - var)

    return results, d_vars
