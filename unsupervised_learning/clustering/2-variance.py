#!/usr/bin/env python3
"""Module that calculates total intra-cluster variance for a dataset."""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set.

    Args:
        X: numpy.ndarray of shape (n, d) containing the data set
        C: numpy.ndarray of shape (k, d) containing centroid means

    Returns:
        var: total variance, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # Calculate squared Euclidean distances between points and centroids
    # Shape of distances: (n, k)
    distances = np.square(X[:, np.newaxis] - C).sum(axis=2)

    # Minimum squared distance to any centroid for each point
    min_distances = np.min(distances, axis=1)

    # Sum of minimum squared distances
    return np.sum(min_distances)
