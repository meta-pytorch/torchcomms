# Copyright 2017-2023 Lawrence Livermore National Security, LLC and other
# Hatchet Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import cython
import numpy as np
from libc.math cimport floor


# Description: All cython code related to graphframe.py
@cython.boundscheck(False)
def insert_one_for_self_nodes(const long snio_len, short[:] self_missing_node, const long[:] snio_indices):
    """Adds a '1' where rows are in self but not in other."""
    for i in range(snio_len):
        self_missing_node[snio_indices[i]] = 1

cpdef fast_not_isin(const unsigned long long[:,:] arr1, const unsigned long long[:,:] arr2, const long arr1_len, const long arr2_len):
    """Check if elements in arr1 exist in arr2.

    A fast check to see if each element in arr1 exists in arr2. This returns a
    vector mask equivalent to what the operation ~df.isin( ... ) would return.

    Arguments:
        arr1 (unsigned long long [][]): The array of values we are searching for.
        arr2 (unsigned long long [][]): The sorted array of values we are searching in.

    Returns:
        (bool []): A boolean mask over arr1 indicating whether each element is or is not in
            the function. True indicates that an element is not in arr2.
    """
    result = np.zeros(len(arr1), dtype=np.bool_)
    cdef long index = -1
    cdef unsigned long long prior = -1
    cdef long i = 0

    for i in range(arr1_len):
        # mini optimization: don't perform binsearch if we already know node is/isnot in graph
        if prior == arr1[i][0]:
            result[arr1[i][1]] = result[arr1[i-1][1]]
        else:
            index = binary_search(arr2, arr2_len, arr1[i][0])
            if index == -1:
                result[arr1[i][1]] = True
            else:
                result[arr1[i][1]] = False

    prior = arr1[i][0]

    return result

cdef binary_search(const unsigned long long[:,:] array, const long arr_len, const unsigned long long key):
    """Basic Binary Search.

    Arguments:
        array (unsigned long long [][]): Array we are searching in (must be sorted).

    Returns: The index of key if found, -1 if not found.
    """
    cdef long L = 0
    cdef long R = arr_len - 1
    cdef long midpt = 0
    while (L <= R):
        midpt = <long>floor((L+R)/2)
        if array[midpt][0] < key:
            L = midpt + 1
        elif array[midpt][0] > key:
            R = midpt - 1
        else:
            return midpt
    return -1
