#!/usr/bin/env python3
"""
evaluate the neuron
"""


import numpy as np


class Neuron():
    """
    neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Class constructor for Neuron
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function for Weights
        """
        return self.__W

    @property
    def b(self):
        """
        getter function for bias
        """
        return self.__b

    @property
    def A(self):
        """
        getter function for activation
        """
        return self.__A

    def forward_prop(self, X):
        """
        Forward propagation function of the neuron
        """
        z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.round(A).astype(int)
        return prediction, cost
