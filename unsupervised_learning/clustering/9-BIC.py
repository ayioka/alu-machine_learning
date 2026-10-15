#!/usr/bin/env python3
"""BIC for finding the best number of GMM clusters"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC

    Returns: best_k, best_result, l, b or None, None, None, None
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, (int, float)) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None
    l = []
    b = []
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        p = k * d + k * d * (d + 1) / 2 + k - 1
        l.append(ll)
        b.append(p * np.log(n) - 2 * ll)
        results.append((pi, m, S))
    l = np.array(l)
    b = np.array(b)
    best = np.argmin(b)
    return kmin + best, results[best], l, b
