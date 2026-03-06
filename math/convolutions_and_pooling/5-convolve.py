#!/usr/bin/env python3
"""
defines a function that performs a convolution on images
using multiple kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    function that performs a convolution on images using multiple kernels
    with custom padding and stride
    Returns: numpy.ndarray - shape (m, new_h, new_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
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
       images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant'
       )

    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, nc))

    for i in range(output_h):
        for j in range(output_w):
            for k in range(nc):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw
                output[:, i, j, k] = np.sum(
                    padded_images[
                       :, vert_start:vert_end, horiz_start:horiz_end, :
                    ] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
