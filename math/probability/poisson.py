#!/usr/bin/env python3
'''
poisson class that represents a poisson distribution
'''


class Poisson:
    '''
    poisson class with pmf and cdf functions
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        '''
        pmf function returns kth value of poisson distribution
        '''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i
        return (self.lambtha**k) * (2.7182818285**-self.lambtha) / k_fact

    def cdf(self, k):
        '''
        cdf function returns the sum of the kth value of poisson distribution
        '''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(n) for n in range(k + 1)])
