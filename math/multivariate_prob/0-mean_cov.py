#!/usr/bin/env python3
'''
mean and cov functions
'''
import numpy as np


def mean_cov(X):
    """
    Function that calculates the mean and cov of a data set
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0).reshape(1, d)
    X = X - mean
    cov = np.dot(X.T, X) / (n - 1)
    return mean, cov
