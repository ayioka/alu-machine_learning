#!/usr/bin/env python3
"""Find the best number of clusters for a GMM using BIC."""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=100, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0 or kmin > kmax:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    bics = []
    log_likes = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l_g = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        log_likes.append(l_g)

        # Number of parameters p:
        # (k - 1) priors + k * d means + k * d * (d + 1) / 2 covariance values
        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)

        bic = p * np.log(n) - 2 * l_g
        bics.append(bic)

    bics = np.array(bics)
    log_likes = np.array(log_likes)

    best_k_idx = np.argmin(bics)
    best_k = kmin + best_k_idx

    # Re-run EM for optimal k
    best_result = expectation_maximization(
        X, best_k, iterations, tol, verbose)

    return best_k, best_result, bics, log_likes
