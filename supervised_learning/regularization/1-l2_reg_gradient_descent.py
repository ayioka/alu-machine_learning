#!/usr/bin/env python3
"""L2 Gradient Descent"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates w and b of NN using gradient descent L2
    Y: shape(classs, m)
    w: dict. of w and b of NN
    cache: dict. of outputs of each layer of NN
    alpha: Learning rate
    lambtha: L2 regularisation parameter
    L: no of layers
    activations: tanh except last(softmax)
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        """
        Loops over layers decrementing
        """
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = dA * (1 - (A ** 2))
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + ((lambtha / m) * W)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
    return weights
