#!/usr/bin/env python3
"""Total intra-cluster variance"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance

    Returns: var or None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if C.shape[1] != X.shape[1]:
        return None
    D = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.sum(np.min(D, axis=1))
