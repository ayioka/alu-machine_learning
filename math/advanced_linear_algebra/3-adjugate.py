#!/usr/bin/env python3
"""
defines djugate(matrix) function
"""


def adjugate(matrix):
    """
    Calculates the adjugate of a square matrix.
    """
    if not isinstance(matrix, list) or len(matrix) == 0 \
       or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactors = []
    for i in range(n):
        cofactors.append([])
        for j in range(n):
            submatrix = [row[:j] + row[j + 1:] for
                         row in (matrix[:i] + matrix[i + 1:])]
            sign = (-1) ** (i + j)
            cofactors[i].append(sign * determinant(submatrix))

    adjugate_matrix = [[cofactors[j][i] for
                        j in range(n)] for i in range(n)]

    return adjugate_matrix


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
        submatrix = [row[:i] + row[i+1:] for
                     row in matrix[1:]]
        sign = (-1)**i
        det += sign * matrix[0][i] * determinant(submatrix)

    return det


def transpose(matrix):
    """
    Calculates the transpose of a matrix.
    """
    n = len(matrix)
    m = len(matrix[0])
    transpose_matrix = [[0 for _ in range(n)] for _ in range(m)]

    for i in range(n):
        for j in range(m):
            transpose_matrix[j][i] = matrix[i][j]

    return transpose_matrix
