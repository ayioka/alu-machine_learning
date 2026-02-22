#!/usr/bin/env python3
'''
expoential class that represents an exponential distribution
'''


class Exponential:
    '''
    Exponential class with pdf function
    '''
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)
        if data is None:
            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        '''
        df function returns xth value of exponential distr
        '''
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        '''
        cdf function returns the sum of the xth value of exponential
        '''
        if x < 0:
            return 0
        return 1 - 2.7182818285**(-self.lambtha * x)
