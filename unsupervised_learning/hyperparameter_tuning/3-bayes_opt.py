#!/usr/bin/env python3
"""Module for Bayesian Optimization initialization."""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize Bayesian Optimization.

        Args:
            f: the black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1), sampled inputs
            Y_init: numpy.ndarray of shape (t, 1), sampled outputs
            bounds: tuple (min, max) for the search space
            ac_samples: number of acquisition sample points
            l: length parameter for the kernel
            sigma_f: standard deviation of the black-box function output
            xsi: exploration-exploitation factor
            minimize: True for minimization, False for maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
