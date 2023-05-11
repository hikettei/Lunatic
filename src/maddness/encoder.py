
"""
Here's the implementation of Maddness Encoderh accelarated with Numba.

Encoder Part is consisted of:

  1. Optimizing Encoder Function g(a)
  2. 
  3.

"""

import numpy as np
import numba
import os, resource
from typing import Any, List, Optional, Tuple, Union

from .hash_function_helper import (
    create_bucket_toplevel
    )

def train_encoder(A_offline: np.ndarray,
                  C:int = 16,
                  nsplits:int = 4,
                  verbose=False,
                  optimize_prototypes=True):
    """
    The function train_encoder obtains following parameters by A_offline:
    1. Binary-Hasing-Tree for each subspace.
    2. Each Bucket's parameter (alpha, beta, split-dim, threshold).

    Also. ~~~

    Input:
        A_offline

    Output:
        pass
    """
    _learn_binary_tree_splits(A_offline, 4)


def _learn_binary_tree_splits(subspace: np.ndarray,
                             nsplits:int):
    """

    Parameters:
      nsplits - The depth of binary-tree

    Return:
     (Buckets, Loss)
    """

    # aux
    K = 2**nsplits
    N, STEP = subspace.shape

    subspace = subspace.astype(np.float32)
    
    binary_tree_top  = create_bucket_toplevel(N)
    col_losses      = np.zeros([STEP], np.float32)


    for nth in range(nsplits):
        col_losses.fill(0.0)
        binary_tree_top.sumup_col_sum_sqs(col_losses, subspace)

        # Determines a strategy for which rows to start it.
        try_dim_order = np.argsort(col_losses)[::-1]

        col_losses.fill(0.0)

        for dth, dim in enumerate(try_dim_order):
            early_stopping_p = binary_tree_top.optimal_val_splits(subspace, col_losses, dim, dth)
            if early_stopping_p:
                break

        # Optimizing Binary-Tree...
        best_trying_dim_idx  = np.argmin(col_losses)
        best_candidate_idx = try_dim_order[best_trying_dim_idx]

        binary_tree_top.optimize_thresholds(subspace, best_candidate_idx, best_trying_dim_idx)
        
        binary_tree_top.optimize_splits(subspace, best_trying_dim_idx, nth)

    # Compute Loss

    return binary_tree_top
