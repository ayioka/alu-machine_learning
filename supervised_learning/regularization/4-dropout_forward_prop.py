#!/usr/bin/env python3
"""Importing tensorflow"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using dropout
    X: np array shape(nx, m)
    keep_prob: probability node will be kept
    layers to use softmax
    return: Dixt. containing output of eachlayer
    and dropot mask
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        A = cache['A' + str(i)]
        W = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        Z = np.matmul(W, A) + b
        if i == L - 1:
            t = np.exp(Z)
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            A /= keep_prob
            cache['D' + str(i + 1)] = D
        cache['A' + str(i + 1)] = A
    return cache
