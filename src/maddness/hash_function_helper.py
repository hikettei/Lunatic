
"""
hash_function_helper.py

"""

import numpy as np
import numba
import os, resource
from typing import Any, List, Optional, Tuple, Union

from utils import cumulative_sses


class Bucket:
    """
4.1 Hash Function Family g(a)
 - MaddnessHash (BinaryTree)
 - 4.2 Learning Hash-Function Parameters
   Let be B(t, i) the bucket which is helper structure where t is the tree's depth and is in the index in the node:

Figure:
              B(1, 1)                  | nth=0
         /----------------\            |
     B(2, 1)            B(2,2)         | nth=1
   /---------\        /---------\      |
B(3, 1)  B(3, 2)   B(3, 3)  B(3, 4)    | nth=2
                                       | ...
                                       | nth=nsplits

    Each bucket has these parameters:

      1. indices   (List) Index of columns, which classified to current Bucket.
      2. split_dim (int) 
      3. threshold (single_float)
      4. alpha     (single_flaot)
      5. beta      (single_float)

    Add: print-bucket
    """
    def __init__(self, indices, tree_level, idx):
        self.indices    = indices
        
        self.N = len(self.indices)
        
        # [first trying, second trying, ...]
        # Note: The order indices doesn't correspond with subspace's each rows.
        self.threshold_candidates = []
        
        self.split_dim = 0
        self.threshold = 0
        self.alpha = 0
        self.beta  = 0

        self.tree_level = tree_level
        self.idx = idx

        self.right_node = None
        self.left_node  = None

    def optimal_val_splits(self, subspace, loss_out, split_dim, dth):
        """
        Tests for the **all nodes** belonging to the bucket, the given split-dim to find out one minimizing loss.

        The result is stored in self.threshold_candidates, being restored like: self.threshold_candidates[dth]

        Input:
          subspace  - the subspace the bucket working on.
          loss_out  - the np.nparray matrix to save the loss[d].
          split-dim - the axis for the function to test optimal_val_splits.
          dth       - whichth trying is it?

        Output:
          early_stopping_p - If True, means the function has already found out the best split-dim.
        """

        def compute_on(bucket):
            pass

        def explore_bucket(bucket):
            if bucket is not None:
                return 0
            return None

        # Explore: the current bucket itself, and right/left side nodes.
        result1 = compute_on(self)
        result2 = explore_bucket(self.right_node, subspace)
        result3 = explore_bucket(self.left_node,  subspace)

        return result1 or result2 or result3
        

    def right_idx(self):
        pass

    def left_idx(self):
        pass
    
    def splits(self):
        pass

    def flatten(self):
        pass

    def col_variances(self, original_mat):
        """
        The method col_variances find the variances among rows, returning [1, D] matrix storing the result.

        Input:
            original_mat - the subspace corresponds with the buckets.
        """
        if self.N < 1:
            # subspace   variances
            #   ++--
            #   ++--  =>   ++--
            #   ++--
            return np.zeros(original_mat.shape[1], np.float32)

        jurisdictions = np.sum(original_mat[self.indices], axis=0)

        sumX2 = np.square(jurisdictions)
        sumX  = jurisdictions

        E_X2  = sumX2 / self.N
        E_X   = sumX  / self.N

        ret   = E_X2 - (E_X * E_X)

        return np.maximum(0, ret)

    def sumup_col_sum_sqs(self, out, original_mat):
        """
        Sum up the result of col_variances of current bucket and nodes belonging to the bucket into out.

        Side Effects:
            out - Fullfilled with result. (First, fill it with 0.0)

        Input:
            out          - np.ndarray matrix to be overwritten
            original_mat - the subspace corresponds to the bucket-tree.
        Return:
            nil
        """
        out += (self.col_variances(original_mat) * self.N)

        if self.right_node is not None:
            self.right_node.sumup_col_sum_sqs(out, original_mat)

        if self.left_node is not None:
            self.left_node.sumup_col_sum_sqs(out, original_mat)

        return None
        
def create_bucket_toplevel(N):
    """
    The function create_bucket_toplevel creates Bucket as top_level.
    To branch nodes under top-level-bucket, call this function: bucket.split
    """
    return Bucket(np.arange(N), 0, 0)

def _compute_optimal_val_splits(bucket, subspace, dim):
    """
    See the original paper for details:
        Appendix C, Algorithm 3, Optimal Split Threshold Within a Bucket.

    Input:
      bucket   - as it is.
      subspace - as it is.
      dim      - as it is.

    Output:
       values - (use-split-val, use-loss)
    """

    ## If bucket.N < 2, can't compute optimal_val_splits, moreover, it also indicates that the bucket is working well.
    if bucket.N < 2:
        return (0.0, 0.0)

    N, _ = subspace.shape

    # Sort by [:, dim].
    sort_idx     = np.argsort(X[:, dim]) #'<
    sort_idx_rev = sort_idx[::-1]

    X_sort     = subspace[sort_idx, :]
    X_sort_rev = subspace[sort_idx_rev, :]

    # Compute SSE
    sses_heads = cumulative_sses(X_sort)
    sses_tails = cumulative_sses(X_sort_rev)[::-1]

    
    sses_heads += sses_tails

    sses_out = np.sum(sses_heads, axis=1)

    # Find out the best_idx, and second_idx

    best_idx = np.argmin(sses)
    next_idx = min(N - 1, best_idx + 1)

    col = sort_idx[:, dim]

    best_val = (col[sort_idx[best_idx]] + col[sort_idx[next_idx]]) / 2

    return best_val, sses[best_idx]