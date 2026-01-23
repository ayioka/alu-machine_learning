#!/usr/bin/env python3
'''defines function that adds two arrays element-wise'''


def add_arrays(arr1, arr2):
    '''returns new list that is the sum of two arrays '''
    if len(arr1) != len(arr2):
        return None
    added_arr = []
    for i in range(len(arr1)):
        added_arr.append(arr1[i] + arr2[i])
    return added_arr
