#!/usr/bin/env python3
"""Importing numpy"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights of NN with Dropout using gradient descent
    Y:npshape(classes, m)
    alpha: learning rate
    keep_prob: probability node to be kept
    """
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            # For ouput layer
            dZ = A - Y
        else:
            # For hidden layers(tanhz)
            dZ = dA * (1 - (A ** 2))
            # Apply dropout to gradient element-wise
            dZ = np.multiply(dZ, cache['D' + str(i)])
            # Rescale gradients to account for dropped neurons
            dZ /= keep_prob
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        weights['W' + str(i)] = W - (alpha * dW)
        weights['b' + str(i)] = b - (alpha * db)
    return weights
