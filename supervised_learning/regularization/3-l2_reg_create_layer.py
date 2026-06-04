#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer including l2 reg
    prev: tensor with output of prev layer
    n: no of nodes
    activation: activation function
    lambtha: L2 regularization parameter
    Returns: output of new layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel,
                            kernel_regularizer=l2)
    return layer(prev)
