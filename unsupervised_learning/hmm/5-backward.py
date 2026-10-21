#!/usr/bin/env python3
"""Module that performs the backward algorithm for an HMM"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden markov model

    Observation: numpy.ndarray of shape (T,), observation indices
    Emission: numpy.ndarray of shape (N, M), emission probabilities
    Transition: numpy.ndarray of shape (N, N), transition probabilities
    Initial: numpy.ndarray of shape (N, 1), starting probabilities
    Returns: P, B, or None, None on failure
        P is the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) of backward probabilities
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    N, M = Emission.shape
    if not isinstance(Transition, np.ndarray) or \
            Transition.shape != (N, N):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    T = Observation.shape[0]
    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[:, t + 1] * Transition[i, :] *
                             Emission[:, Observation[t + 1]])
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B
