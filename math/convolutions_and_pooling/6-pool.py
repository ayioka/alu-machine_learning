#!/usr/bin/env python3
"""
defines a function that performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images
    with custom kernel shape and stride
    Returns: numpy.ndarray - shape (m, new_h, new_w, c)
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    pooled_images = np.zeros((m, out_h, out_w, c))

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw

            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(
                    images[:, h_start:h_end, w_start:w_end, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(
                    images[:, h_start:h_end, w_start:w_end, :],
                    axis=(1, 2)
                )

    return pooled_images
