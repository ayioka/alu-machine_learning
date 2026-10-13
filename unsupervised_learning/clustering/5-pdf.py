#!/usr/bin/env python3
"""Calculate the probability density function of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function of a Gaussian distribution."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    det = np.linalg.det(S)
    if det <= 0:
        return None

    inv = np.linalg.inv(S)
    diff = X - m

    # Calculate exponent term without loops: sum over axis 1 of (diff @ inv) * diff
    exponent = -0.5 * np.sum(np.matmul(diff, inv) * diff, axis=1)
    norm_const = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))

    P = norm_const * np.exp(exponent)
    P = np.maximum(P, 1e-300)

    return P
