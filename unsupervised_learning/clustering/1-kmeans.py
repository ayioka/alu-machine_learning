#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset

    Returns: C, clss or None, None on failure
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    d = X.shape[1]
    low = X.min(axis=0)
    high = X.max(axis=0)
    C = np.random.uniform(low, high, (k, d))
    for i in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=2), axis=1)
        new_C = C.copy()
        for j in range(k):
            pts = X[clss == j]
            if pts.shape[0] == 0:
                new_C[j] = np.random.uniform(low, high)
            else:
                new_C[j] = pts.mean(axis=0)
        if np.array_equal(new_C, C):
            break
        C = new_C
    clss = np.argmin(np.linalg.norm(X[:, None] - C, axis=2), axis=1)
    return C, clss
