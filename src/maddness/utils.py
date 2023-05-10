
"""
Utils functions for maddness.
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
    
    Input:
      X - [N D] where dtype = single-float
    Output
      cumsses - [N D] where dtype = single-float
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
        one_over_count = 1.0 / (i + 1) # (1-decay)
        for j in range(D):
            cumX_column[j] += X[i, j]
            cumX2_column[j] += X[i, j] * X[i, j]
            meanX = cumX_column[j] * one_over_count
            cumsses[i, j] = cumX2_column[j] - (cumX_column[j] * meanX)
    return cumsses

@numba.jit('u1[:, :](u1[:, :], int64)')
def sparsify_encoded_A(A_encoded, K):
    """
    The function sparsify_encoded_A sparsify A_encoded.
    For Example: Let be C 4.
      [2] is encoded into [0, 0, 1, 0].
    Input:
        A_encoded [N, C] but dtype = '(signed-byte 8)
    Return:
        One-hot encoded matrix [N, C * K], dtype = '(signed-byte 8)
    """
    N, C = A_encoded.shape
    D    = C * K

    out = np.array([N, D], np.int8)

    for n in range(N):
        for c in range(C):
            code_left = A_encoded[n, c]
            dim_left  = (K * c) + code_left
            out[n, dim_left] = 1
    return out

