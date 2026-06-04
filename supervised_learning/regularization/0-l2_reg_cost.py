#!/usr/bin/env python3
"""L2 regularisation"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates cost of NN with L2 regularisation
    cost: cost without L2 regularization
    weights: dict. of weights n biases
    L: no of layers  of NN
    m: no of data points used
    Returns: Cost accounting L2
    """
    weighted_sum = 0
    # Loop over each layer
    for layer_index in range(1, L + 1):
        # add squared norm(sum of squares)
        weighted_sum += np.linalg.norm(weights['W' + str(layer_index)]) ** 2
    # Calculate regularisation term
    reg_term = lambtha / (2 * m) * weighted_sum
    # Add term to original cost
    Jreg = cost + reg_term
    return Jreg
