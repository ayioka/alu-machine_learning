#!/usr/bin/env python3
"""Importing modules"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    Creates a layer using dropout
    return: output of new layer
    """
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel)
    output = tf.layers.Dropout(keep_prob)
    return output(layer(prev))
