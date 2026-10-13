#!/usr/bin/env python3
"""Perform PCA on a dataset to maintain a fraction of variance."""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset X to retain a specified variance.

    Parameters:
        X: numpy.ndarray of shape (n, d) where n is the number of data points,
           d is the original dimensionality, and mean across axis=0 is 0.
        var: fraction of variance to maintain.

    Returns:
        W: weights matrix of shape (d, nd) where nd is the new dimensionality.
    """
    # Compute Singular Value Decomposition (SVD)
    # X = U * S * Vh
    _, s, vh = np.linalg.svd(X, full_matrices=False)

    # Compute variance explained ratio using cumulative sum
    cum_var = np.cumsum(s ** 2) / np.sum(s ** 2)

    # Find the number of dimensions needed to reach or exceed 'var'
    nd = np.argwhere(cum_var >= var)[0][0] + 1

    # Extract top principal components and transpose to get shape (d, nd)
    W = vh[:nd].T

    return W
