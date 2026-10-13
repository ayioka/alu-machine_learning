#!/usr/bin/env python3
"""Module for K-means clustering algorithm."""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing maximum iterations

    Returns:
        C, clss or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    initialize = __import__('0-initialize').initialize
    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for i in range(iterations):
        C_copy = C.copy()

        # Calculate distances and assign clusters (no loop)
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids (Loop 1 allowed inside main iteration loop)
        for j in range(k):
            cluster_points = X[clss == j]
            if len(cluster_points) == 0:
                C[j] = np.random.uniform(low, high, size=(1, X.shape[1]))
            else:
                C[j] = cluster_points.mean(axis=0)

        # Update cluster assignments after centroid movement
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Check early stopping condition
        if (C == C_copy).all():
            break

    return C, clss
