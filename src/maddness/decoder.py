
import numpy as np
from .cffi_utils import (
    maddness_scan
)

# Original Repo: 

def maddness_quantize_luts(luts: np.ndarray, force_power_of_2: bool = True):
    mins = luts.min(axis=(0, 2))
    maxs = luts.max(axis=(0, 2))

    gaps = maxs - mins
    gap = np.max(gaps)
    if force_power_of_2:
        exponent = np.ceil(np.log2(gap))
        scale = 2 ** int(-exponent)  # scale is a power of 2, so can just shift
        scale *= 255.5 - 1e-10  # so max val is at most 255
    else:
        scale = (255.5 - 1e-10) / gap

    offsets = mins[np.newaxis, :, np.newaxis]
    luts_quantized = (luts - offsets) * scale
    luts_quantized = (luts_quantized + 0.5).astype(np.uint8)

    assert np.min(luts_quantized) >= 0
    assert np.max(luts_quantized) <= 255.0

    return luts_quantized, offsets.sum(), scale

def maddness_lut(q: np.ndarray, all_prototypes: np.ndarray) -> np.ndarray:
    q = q.reshape(1, 1, -1)  # all_prototypes is shape C, K, D
    return (q * all_prototypes).sum(axis=2)  # C, K

def construct_lut(B, protos, C, nsplits):
    """

    Return:
       luts [M C K]
       scale
       offset
    """
    B = np.atleast_2d(B)
    K = 2**nsplits
    luts = np.zeros((B.shape[0], C, K))

    for i, q in enumerate(B):
        luts[i] = maddness_lut(q, protos)
    luts, offset, scale = maddness_quantize_luts(luts)
    # Luts ... [M C K]
    return luts, scale, offset

def lut_scan(A_enc, C, N, luts):
    return maddness_scan(A_enc, C, N, luts)
