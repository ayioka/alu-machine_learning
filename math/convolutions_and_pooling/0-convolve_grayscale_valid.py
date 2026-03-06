#!/usr/bin/env python3
"""
defines a function that performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function to perform a valid convolution on grayscale images
    Returns:numpy.ndarray - shape (m, new_h, new_w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    new_h = h - kh + 1
    new_w = w - kw + 1

    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(
                 images[:, i:i+kh, j:j+kw] * kernel,
                 axis=(1, 2)
                 )

    return output
