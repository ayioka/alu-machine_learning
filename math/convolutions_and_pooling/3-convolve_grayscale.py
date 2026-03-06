#!/usr/bin/env python3
"""
defines a function that performs a convolution on grayscale images(strides)
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on grayscale images
    with custom padding and stride
    Returns: numpy.ndarray - shape (m, new_h, new_w)
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
    )

    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2)
            )

    return output
