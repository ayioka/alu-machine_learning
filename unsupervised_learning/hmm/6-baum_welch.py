#!/usr/bin/env python3
"""Module that performs the Baum-Welch algorithm for an HMM"""
import numpy as np


def _forward(Observations, Transition, Emission, Initial):
    """Computes the forward probabilities, shape (M, T)"""
    M = Transition.shape[0]
    T = Observations.shape[0]
    F = np.zeros((M, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
    for t in range(1, T):
        F[:, t] = np.matmul(F[:, t - 1], Transition) * \
            Emission[:, Observations[t]]
    return F


def _backward(Observations, Transition, Emission):
    """Computes the backward probabilities, shape (M, T)"""
    M = Transition.shape[0]
    T = Observations.shape[0]
    B = np.zeros((M, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        B[:, t] = np.matmul(Transition,
                            Emission[:, Observations[t + 1]] * B[:, t + 1])
    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden markov model

    Observations: numpy.ndarray of shape (T,), observation indices
    Transition: numpy.ndarray of shape (M, M), initial transitions
    Emission: numpy.ndarray of shape (M, N), initial emissions
    Initial: numpy.ndarray of shape (M, 1), starting probabilities
    iterations: number of expectation-maximization steps
    Returns: the converged Transition, Emission, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    M = Transition.shape[0]
    if Transition.shape != (M, M):
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or \
            Emission.shape[0] != M:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (M, 1):
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    N = Emission.shape[1]
    T = Observations.shape[0]
    Transition = Transition.copy()
    Emission = Emission.copy()
    for _ in range(iterations):
        F = _forward(Observations, Transition, Emission, Initial)
        B = _backward(Observations, Transition, Emission)
        P = np.sum(F[:, -1])
        # xi[i, j, t]: prob of state i at t and state j at t + 1
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            xi[:, :, t] = F[:, t].reshape(-1, 1) * Transition * \
                (Emission[:, Observations[t + 1]] * B[:, t + 1]) / P
        # gamma[i, t]: prob of state i at t
        gamma = F * B / P
        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :-1], axis=1).reshape(-1, 1)
        denom = np.sum(gamma, axis=1).reshape(-1, 1)
        for k in range(N):
            Emission[:, k] = np.sum(gamma[:, Observations == k], axis=1)
        Emission = Emission / denom
    return Transition, Emission
