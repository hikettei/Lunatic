
"""
hash_function_helper.py

"""

import numpy as np
import numba
import os, resource
from typing import Any, List, Optional, Tuple, Union

from .utils import (
    cumulative_sses,
    learn_quantized_param
)


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
        self.indices = indices
        
        self.N = len(self.indices)
        
        # [first trying, second trying, ...]
        # Note: The order indices doesn't correspond with subspace's each rows.
        self.threshold_candidates = []
        
        self.split_dim = 0
        self.threshold = 0.0
        self.threshold_q = 0
        self.alpha = 0.0
        self.beta  = 0.0

        self.tree_level = tree_level
        self.idx = idx

        self.right_node = None
        self.left_node  = None

    def optimize_thresholds(self, subspace, best_candidate_idx, dim):
        """

        """
        self.split_dim = dim
        self.threshold = self.threshold_candidates[best_candidate_idx]
        self.threshold_q, self.alpha, self.beta = learn_quantized_param(self, subspace, dim)
        ## Reset Params
        self.threshold_candidates = []

        if self.right_node is not None:
            self.right_node.optimize_thresholds(subspace, best_candidate_idx, dim)
        if self.left_node is not None:
            self.left_node.optimize_thresholds(subspace, best_candidate_idx, dim)

        return None

    def optimize_splits(self, subspace, dim, nth_split):
        """

        """

        def explore(bucket):
            if bucket.tree_level <= nth_split:
                bucket.optimize_splits(subspace, dim, nth_split)

        def update_indices(left_indices, right_indices):
            if self.left_node is None and self.right_node is None:
                self.left_node = Bucket(left_indices, 1 + self.tree_level, self.left_idx())
                self.right_node = Bucket(left_indices, 1 + self.tree_level, self.right_idx())
            else:
                self.left_node.indices = left_indices
                self.right_node.indices = right_indices
                self.left_node.N = len(left_indices)
                self.right_node.N = len(right_indices)
                
                explore(self.left_node)
                explore(self.right_node)

        if self.N < 2:
            # Copy of this bucket + Empty
            update_indices(self.indices, [])
        else:
            indices = np.asarray(self.indices)

            jurisdictions = subspace[indices, :]

            mask = jurisdictions[:, dim] > self.threshold
            not_mask = ~mask
            
            left_ids  = indices[not_mask]
            right_ids = indices[mask]
            update_indices(list(left_ids), list(right_ids))
        return None
    
        
        
    
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
            best_val, best_loss = _compute_optimal_val_splits(bucket, subspace, split_dim)

            loss_out[split_dim] += best_loss

            # Storing results.
            bucket.threshold_candidates.append(best_val)

            # Judge early_stopping_p
            if dth > 0 and loss_out[split_dim] >= np.min(loss_out[:dth]):
                return True
            else:
                return False
            

        def explore_bucket(bucket):
            if bucket is None:
                return False
            else:
                bucket.optimal_val_splits(subspace, loss_out, split_dim, dth)

        # Explore: the current bucket itself, and right/left side nodes.
        result1 = compute_on(self)
        result2 = explore_bucket(self.right_node)
        result3 = explore_bucket(self.left_node)

        return result1 and result2 and result3
        

    def right_idx(self):
        return self.idx * 2 + 1
    
    def left_idx(self):
        return self.idx * 2

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

    def update_centroids(self, c, idxs, centroid, all_protos, X_error, X_orig):
        """

        """
        if len(self.indices):
            centroid.fill(0.0)
            centroid[idxs] = X_orig[np.asarray(self.indices)].sum(axis=0) / max(1,  self.N)

            X_error[np.asarray(self.indices)] -= centroid
            all_protos[c, self.idx] = centroid

            if self.left_node is not None and self.right_node is not None:

                self.left_node.update_centroids(c, idxs, centroid, all_protos, X_error, X_orig)
                self.right_node.update_centroids(c, idxs, centroid, all_protos, X_error, X_orig)
        
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
    sort_idx     = np.argsort(subspace[:, dim]) #'<
    sort_idx_rev = sort_idx[::-1]

    X_sort     = subspace[sort_idx, :]
    X_sort_rev = subspace[sort_idx_rev, :]

    # Compute SSE
    sses_heads = cumulative_sses(X_sort)
    sses_tails = cumulative_sses(X_sort_rev)[::-1]

    
    sses_heads += sses_tails

    sses = np.sum(sses_heads, axis=1)

    # Find out the best_idx, and second_idx

    best_idx = np.argmin(sses)
    next_idx = min(N - 1, best_idx + 1)

    col = subspace[:, dim]

    best_val = (col[sort_idx[best_idx]] + col[sort_idx[next_idx]]) / 2

    return best_val, sses[best_idx]

# Fix it.
def flatten_buckets1(buckets: List, nsplits: int):
    """

    """
    
    buckets_per_subspace = 0
    for i in range(nsplits):
        buckets_per_subspace += 2**i

    total_buckets = buckets_per_subspace * len(buckets)
    

    # (size of protos, buckets_per_subspace)
    split_dim = np.zeros(total_buckets, np.int32).reshape((-1, buckets_per_subspace))
    threshold = np.zeros(total_buckets, np.int8).reshape((-1, buckets_per_subspace))
    alpha = np.zeros(total_buckets, np.float32).reshape((-1, buckets_per_subspace))
    beta = np.zeros(total_buckets, np.float32).reshape((-1, buckets_per_subspace))

    def gather_buckets(bucks, depth=0):
        for i, b in enumerate(bucks):
            if b is None:
                return
            n = b.idx + 2**depth -1
            split_dim[i, n] = b.split_dim
            threshold[i, n] = b.threshold_q
            alpha[i, n]     = b.alpha
            beta[i, n]      = b.beta

        if depth+1 == nsplits:
            return
            
        gather_buckets([b.left_node for b in bucks], depth=depth+1)
        gather_buckets([b.right_node for b in bucks], depth=depth+1)
        
    gather_buckets(buckets)
    return split_dim.reshape(-1), threshold.reshape(-1), alpha.reshape(-1), beta.reshape(-1)

def flatten_buckets(buckets: List, nsplits: int):
    """

    """
    
    buckets_per_subspace = 2**nsplits
    total_buckets = buckets_per_subspace * len(buckets)
    

    # (size of protos, buckets_per_subspace)
    split_dim = np.zeros(total_buckets, np.uint32)
    threshold = np.zeros(total_buckets, np.int8)
    alpha = np.zeros(total_buckets, np.float32)
    beta = np.zeros(total_buckets, np.float32)
    
    def gather_buckets(buck, total_offset, local_offset=0):
        split_dim[total_offset + local_offset] = buck.split_dim
        threshold[total_offset + local_offset] = buck.threshold_q
        alpha[total_offset + local_offset] = buck.alpha
        beta[total_offset + local_offset] = buck.beta
        

        def explore(sub_bucket):
            if sub_bucket is not None:
                gather_buckets(sub_bucket, total_offset, local_offset=sub_bucket.idx)
                
        explore(buck.left_node)
        explore(buck.right_node)

    [gather_buckets(buck, i*buckets_per_subspace) for i, buck in enumerate(buckets)]

    return split_dim, threshold, alpha, beta
