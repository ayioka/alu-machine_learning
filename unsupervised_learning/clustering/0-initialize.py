#!/usr/bin/env python3
"""Initialize K-means cluster centroids."""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means using a

    multivariate uniform distribution.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    d = X.shape[1]

    centroids = np.random.uniform(low, high, size=(k, d))
    return centroids
