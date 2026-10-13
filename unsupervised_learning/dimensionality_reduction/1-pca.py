#!/usr/bin/env python3
"""Perform PCA on a dataset to reduce to a specific number of dimensions."""
import numpy as np


def pca(X, ndim):
    """Transforms a dataset X into a new dimensionality ndim using PCA.

    Parameters:
        X: numpy.ndarray of shape (n, d) where n is the number of data points
           and d is the original dimensionality.
        ndim: target number of dimensions (int).

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the transformed dataset.
    """
    # Mean-center the dataset
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Compute Singular Value Decomposition (SVD)
    _, _, vh = np.linalg.svd(X_centered, full_matrices=False)

    # Extract weight matrix for top ndim components: shape (d, ndim)
    W = vh[:ndim].T

    # Transform dataset to lower-dimensional space
    T = np.matmul(X_centered, W)

    return T
