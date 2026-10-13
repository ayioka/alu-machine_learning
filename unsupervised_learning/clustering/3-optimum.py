#!/usr/bin/env python3
"""Module that tests for the optimum number of clusters by variance."""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        kmin: positive integer containing minimum number of clusters to check
        kmax: positive integer containing maximum number of clusters to check
        iterations: positive integer containing maximum iterations for K-means

    Returns:
        results, d_vars or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    kmeans = __import__('1-kmeans').kmeans
    variance = __import__('2-variance').variance

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        results.append((C, clss))
        
        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)

    # Calculate delta variance relative to the variance at kmin
    base_variance = variances[0]
    d_vars = [base_variance - var for var in variances]

    return results, d_vars
