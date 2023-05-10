
"""
The Encoder Part of Maddness.

Encoder Part is consisted of:

  1. Optimizing Encoder Function g(a)
  2. 
  3.

"""

import numpy as np
import numba
import os, resource
from typing import Any, List, Optional, Tuple, Union

@numba.jit('f4[:, :](f4[:, :])', fastmath=True, cache=True, parallel=False, nopython=True)
def cumulative_sses(X):
    """
    The function cumulative_sses computes SSE column-wise.
    [[a1 a2],
     [a3 a4],
     [a5 a6]]
    """
    N, D = X.shape
    cumsses      = np.empty((N, D), X.dtype)
    cumX_column  = np.empty(D,  X.dtype)
    cumX2_column = np.empty(D, X.dtype)
    for j in range(D):
        cumX_column[j] = X[0, j]
        cumX2_column[j] = X[0, j] * X[0, j]
        cumsses[0, j] = 0
    for i in range(1, N):
        one_over_count = 1.0 / (i + 1)
        for j in range(D):
            cumX_column[j] += X[i, j]
            cumX2_column[j] += X[i, j] * X[i, j]
            meanX = cumX_column[j] * one_over_count
            cumsses[i, j] = cumX2_column[j] - (cumX_column[j] * meanX)
    return cumsses
