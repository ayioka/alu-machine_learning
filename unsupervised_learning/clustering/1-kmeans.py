#!/usr/bin/env python3
"""Perform K-means clustering on a dataset."""
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    prev_C = None
    for _ in range(iterations):
        # Calculate distances and assign clusters (Loop 1)
        D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(D, axis=1)

        # Check for convergence
        if prev_C is not None and np.all(C == prev_C):
            break
        prev_C = C.copy()

        # Update centroids (Loop 2)
        new_C = np.copy(C)
        for i in range(k):
            if np.any(clss == i):
                new_C[i] = np.mean(X[clss == i], axis=0)
            else:
                new_C[i] = np.random.uniform(low, high, size=(d,))
        C = new_C

    # Final assignment and centroid recalculation to match return state
    D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(D, axis=1)

    return C, clss
