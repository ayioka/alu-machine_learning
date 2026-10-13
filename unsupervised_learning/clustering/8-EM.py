#!/usr/bin/env python3
"""Perform Expectation Maximization for a GMM."""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=100, tol=1e-5, verbose=False):
    """Performs expectation-maximization for a Gaussian Mixture Model."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    l_prev = 0

    for i in range(iterations):
        g, l_g = expectation(X, pi, m, S)
        if g is None or l_g is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations):
            print("Log Likelihood after {} iterations: {}".format(
                i, l_g.round(5)))

        if i > 0 and abs(l_g - l_prev) <= tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, l_g.round(5)))
            return pi, m, S, g, l_g

        l_prev = l_g

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

    g, l_g = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            iterations, l_g.round(5)))

    return pi, m, S, g, l_g
