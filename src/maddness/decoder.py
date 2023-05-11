
import numpy as np
from .cffi_utils import (
    maddness_lut,
    maddness_scan
)

def construct_lut(B, protos, C, nsplits):
    return maddness_lut(B.reshape(B.shape, order="F"),
                        protos.reshape(protos.shape, order="F"),
                        C,
                        2**nsplits)

def lut_scan(A_enc, luts, N, nsplits):
    return maddness_scan(A_enc, nsplits, N, luts)
