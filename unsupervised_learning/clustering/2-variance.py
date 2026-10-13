#!/usr/bin/env python3
"""Calculate total intra-cluster variance for a dataset."""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate squared Euclidean distances to closest centroids without loops
    D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    min_distances = np.min(D, axis=2)
    var = np.sum(min_distances ** 2)

    return var
