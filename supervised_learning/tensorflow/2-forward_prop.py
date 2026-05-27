#!/usr/bin/env python3
"""Forward Propagation"""
import tensorflow as tf
# Import the create_layer function
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Create the forward propagation graph for the neural network.
    x: placeholder for the input data
    layer_sizes: number of nodes in that layer
    activations: activation function for that layer
    Returns: tensor of the final layer's output
    """

    # Ensure that layer_sizes and activations are of the same length
    assert len(layer_sizes) == len(
        activations), "Layer sizes and activations count must be the same."

    prev = x

    for i in range(len(layer_sizes)):
        with tf.variable_scope("layer_" + str(i)):
            prev = create_layer(prev, layer_sizes[i], activations[i])

    return prev
