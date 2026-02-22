#!/usr/bin/env python3
"""Defives a derived of polynomials"""


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial
    poly: list of integers
    returns: list of integers representing the coefficients
    """
    derivative = []
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for i in range(len(poly)-1, 0, -1):

        derivative.append(poly[i]*i)
    return derivative[::-1]
