#!/usr/bin/env python3
"""Module for Gaussian Process implementation."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the Gaussian Process.

        Args:
            X_init: numpy.ndarray of shape (t, 1), sampled inputs
            Y_init: numpy.ndarray of shape (t, 1), sampled outputs
            l: length parameter for the kernel
            sigma_f: standard deviation of the black-box function output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculate the RBF covariance kernel matrix between two matrices.

        Args:
            X1: numpy.ndarray of shape (m, 1)
            X2: numpy.ndarray of shape (n, 1)

        Returns:
            Covariance kernel matrix of shape (m, n)
        """
        sqdist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
            np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))
