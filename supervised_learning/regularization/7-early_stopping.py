#!/usr/bin/env python3
"""Importing libraries"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    cost: current val cost
    opt_cost: lowest recorded val cost
    threshold: used for early stopping
    patience: patience count
    count: how long threshold hasn't been met
    Returns: boolean of whethre network is stopped early
    followed by updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count == patience:
        return True, count

    return False, count
