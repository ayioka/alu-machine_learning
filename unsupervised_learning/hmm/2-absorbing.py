#!/usr/bin/env python3
"""Module that determines whether a Markov chain is absorbing"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing

    P: numpy.ndarray of shape (n, n), the standard transition matrix
    Returns: True if it is absorbing, or False on failure
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape != (n, n) or n < 1:
        return False
    diag = np.diag(P)
    reached = (diag == 1)
    if not np.any(reached):
        return False
    # Walk backwards from absorbing states: a state can reach an
    # absorbing state if it has a path into an already-reached state
    changed = True
    while changed:
        changed = False
        can_reach = np.any(P[:, reached] > 0, axis=1) & ~reached
        if np.any(can_reach):
            reached = reached | can_reach
            changed = True
    return bool(np.all(reached))
