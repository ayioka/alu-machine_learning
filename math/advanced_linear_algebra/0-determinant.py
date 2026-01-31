#!/usr/bin/env python3
"""
Defines a function that Calculates the determinant of a square matrix.
"""


def determinant(matrix):
    """
    matrix: A list of lists representing a square matrix.
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a square matrix")

    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if matrix == [[]]:
            return 1
        if len(row) != len(matrix):
            raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i in range(n):
        submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        sign = (-1)**i
        det += sign * matrix[0][i] * determinant(submatrix)

    return det
