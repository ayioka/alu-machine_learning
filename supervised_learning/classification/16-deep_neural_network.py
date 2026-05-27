#!/usr/bin/env python3
"""Deep Neural Network"""


import numpy as np


class DeepNeuralNetwork:
    '''
    Initialize Deep Neural Network class
    '''
    def __init__(self, nx, layers):
        '''
        Deep Neural Network class constructor
        '''
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
                self.weights['b1'] = np.zeros((layers[i], 1))

            else:
                self.weights['W{}'.format(i + 1)] = np.random.randn(
                    layers[i],
                    layers[i - 1]) * np.sqrt(2 / layers[i - 1])

                self.weights['b{}'.format(
                    i + 1)] = np.zeros((layers[i], 1))
