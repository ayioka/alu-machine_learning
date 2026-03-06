#!/usr/bin/env python3
"""
defines a function that performs a same convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    '''
    Function to perform a same convolution on grayscale images
    '''

    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded_images = np.pad(
            images,
            ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode='constant'
            )

    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:i+kh, j:j+kw]
            convolved_images[:, i, j] = np.sum(
                region * kernel,
                axis=(1, 2)
                )

    return convolved_images
