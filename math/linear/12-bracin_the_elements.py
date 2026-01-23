#!/usr/bin/env python3
"""defines a fuction that performs two matrices"""


def np_elementwise(mat1, mat2):
    """returns: tuple containing element-wise sum, difference, product"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
