#!/usr/bin/env python3
'''
Binomial class that represents a binomial distribution
'''


class Binomial:
    '''
    Binomial class with pmf and cdf functions
    '''
    def __init__(self, data=None, n=1, p=0.5):
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            p = 1.0 - variance / mean
            self.n = round(mean / p)
            self.p = float(mean / self.n)

    def pmf(self, k):
        '''
        pmf function returns the kth value of the binomial distribution
        '''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i
        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i
        nk_fact = 1
        for i in range(1, self.n - k + 1):
            nk_fact *= i

        return (
            (n_fact / (k_fact * nk_fact)) *
            (self.p ** k) * ((1 - self.p) ** (self.n - k))
        )

    def cdf(self, k):
        '''
        cdf function returns the sum of the kth value of the binomial
        '''
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(n) for n in range(k + 1)])
