#!/usr/bin/env python3
"""
defines a function that performs a convolution padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function to perform a convolution on grayscale images with
    custom padding
    Returns: numpy.ndarray - shape (m, new_h, new_w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1

    padded_images = np.pad(
       images,
       ((0, 0), (ph, ph), (pw, pw)),
       mode='constant'
       )

    convolved_images = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel,
                axis=(1, 2)
            )

    return convolved_images
