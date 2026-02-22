#!/usr/bin/env python3
'''
Normal class that represents a normal distribution
'''


class Normal:
    '''
    Normal class with pdf and cdf functions
    '''
    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = (
                sum([(x - self.mean) ** 2 for x in data]) / len(data)
            ) ** 0.5

    def z_score(self, x):
        '''
        z_score function returns the z-score of x
        '''
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        '''
        x_value function returns the x value for a given z-score
        '''
        return self.mean + z * self.stddev

    def pdf(self, x):
        '''
        pdf function returns the PDF value of the normal distribution at x
        '''
        return (
            1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        ) * 2.7182818285 ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        '''
        cdf function returns the CDF value of the normal distribution at x
        '''
        k = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = k - (k ** 3) / 3 + (k ** 5) / 10 - (k ** 7) / 42 + (k ** 9) / 216
        return 0.5 * (1 + (2 / (3.1415926536 ** 0.5)) * erf)
