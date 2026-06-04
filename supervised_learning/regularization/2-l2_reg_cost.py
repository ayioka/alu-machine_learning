#!/usr/bin/env python3
"""Imports tensorflow"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    Caclculates cost of NN with l2
    cost: tensor with cost without l2
    returns: Tensor with cost accounting for l2
    """
    return cost + tf.losses.get_regularization_losses(scope=None)
