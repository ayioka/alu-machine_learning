#!/usr/bin/env python3
"""Module that performs the forward algorithm for an HMM"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden markov model

    Observation: numpy.ndarray of shape (T,), observation indices
    Emission: numpy.ndarray of shape (N, M), emission probabilities
    Transition: numpy.ndarray of shape (N, N), transition probabilities
    Initial: numpy.ndarray of shape (N, 1), starting probabilities
    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) of forward probabilities
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
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = np.sum(F[:, t - 1] * Transition.T *
                         Emission[:, Observation[t]].reshape(-1, 1),
                         axis=1)
    P = np.sum(F[:, -1])
    return P, F
