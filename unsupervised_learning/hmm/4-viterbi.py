#!/usr/bin/env python3
"""Module that performs the Viterbi algorithm for an HMM"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """Calculates the most likely sequence of hidden states for a
    hidden markov model

    Observation: numpy.ndarray of shape (T,), observation indices
    Emission: numpy.ndarray of shape (N, M), emission probabilities
    Transition: numpy.ndarray of shape (N, N), transition probabilities
    Initial: numpy.ndarray of shape (N, 1), starting probabilities
    Returns: path, P, or None, None on failure
        path is a list of length T with the most likely hidden states
        P is the probability of obtaining the path sequence
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
    V = np.zeros((N, T))
    back = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        # scores[i, j]: best prob of being in i at t-1 then moving to j
        scores = V[:, t - 1].reshape(-1, 1) * Transition
        back[:, t] = np.argmax(scores, axis=0)
        V[:, t] = np.max(scores, axis=0) * Emission[:, Observation[t]]
    P = np.max(V[:, -1])
    path = [int(np.argmax(V[:, -1]))]
    for t in range(T - 1, 0, -1):
        path.insert(0, int(back[path[0], t]))
    return path, P
