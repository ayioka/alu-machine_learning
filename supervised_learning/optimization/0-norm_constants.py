#!/usr/bin/env python3
""" Implements normalization_constants
"""
import numpy as np


def normalization_constants(X):
    """Calculates the constants(mean and standard deviation)
    for normalizing features.

    Args:
        X (numpy.ndarray) The input data in the shape (m, nx)
        where m is the number of training examples and nx is
        the number of features.

    Returns:
        tuple: A tuple containing the mean and standard
        deviation for each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std
