#!/usr/bin/env python3
"""Defines a function that multiplies two matrices."""


def mat_mul(mat1, mat2):
    """Returns a new matrix that is the product of two 2D matrices."""

    if len(mat1[0]) != len(mat2):
        return None

    newmat = [[0 for i in range(len(mat2[0]))] for j in range(len(mat1))]

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                newmat[i][j] += mat1[i][k] * mat2[k][j]

    return newmat
