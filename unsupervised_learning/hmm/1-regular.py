#!/usr/bin/env python3
"""Module that computes the steady state of a regular Markov chain"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular
    markov chain

    P: numpy.ndarray of shape (n, n), the transition matrix
    Returns: numpy.ndarray of shape (1, n) containing the steady
    state probabilities, or None on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape != (n, n) or n < 1:
        return None
    if np.any(P < 0) or not np.allclose(np.sum(P, axis=1), 1):
        return None
    # A chain is regular if some power of P is strictly positive;
    # checking up to (n - 1) ** 2 + 1 is sufficient (Wielandt)
    power = P.copy()
    regular_chain = False
    for _ in range((n - 1) ** 2 + 1):
        if np.all(power > 0):
            regular_chain = True
            break
        power = np.matmul(power, P)
    if not regular_chain:
        return None
    # Solve pi (P - I) = 0 with sum(pi) = 1
    A = np.vstack((P.T - np.eye(n), np.ones((1, n))))
    b = np.zeros((n + 1, 1))
    b[-1, 0] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return pi.reshape(1, n)
