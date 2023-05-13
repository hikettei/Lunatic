
import os
from cffi import FFI
import numpy as np

def load_libmithral(libpath="src/maddness/cpp/libMithral"):
    global LIBMITHRAL_STATIC
    global LIBMITHRAL_STATIC1
    ffi = FFI()
    cwd = os.getcwd()
    lib_path = f"{cwd}/{libpath}"

    ffi.cdef("""
void maddness_encode(const float *X,
		     int C,
		     int nsplits,
		     int nrows,
		     int ncols,
		     const uint32_t *splitdims,
		     const int8_t *splitvals,
		     const float *scales,
		     const float *offsets,
		     uint8_t* out);

void mithral_scan_fp32_t(const uint8_t* encoded_mat,
                 int ncodebooks,
                 int M,
                 const uint8_t* luts,
                 uint8_t* out_mat);
    """)
    LIBMITHRAL_STATIC = ffi.dlopen(f"{lib_path}.dylib")
    LIBMITHRAL_STATIC1 = ffi.dlopen("third_parties/cl-xMatrix/source/kernel/libMithral.dylib")
    
    return True

def convert_to_cpp_int(arr):
    ffi = FFI()
    return ffi.cast("int*", arr.ctypes.data)

def convert_to_cpp_uint32(arr):
    ffi = FFI()
    return ffi.cast("uint32_t*", arr.ctypes.data)

def convert_to_cpp_uint8(arr):
    ffi = FFI()
    return ffi.cast("uint8_t*", arr.ctypes.data)

def convert_to_cpp_int8(arr):
    ffi = FFI()
    return ffi.cast("int8_t*", arr.ctypes.data)

def convert_to_cpp_float(arr):
    ffi = FFI()
    return ffi.cast("float*", arr.ctypes.data)

# ncodebooks=16
def maddness_encode(X, STEP, C, nsplits, splitdims, splitvals, scales, offsets, add_offsets=True):
    K = 2**nsplits
    out = np.zeros((X.shape[0], STEP), dtype=np.uint8, order="F")
    LIBMITHRAL_STATIC.maddness_encode(convert_to_cpp_float(X), 
                                      C,
                                      nsplits,
                                      X.shape[0],
                                      X.shape[1],
                                      convert_to_cpp_uint32(splitdims),
                                      convert_to_cpp_int8(splitvals),
                                      convert_to_cpp_float(scales),
                                      convert_to_cpp_float(offsets),
                                      convert_to_cpp_uint8(out))
    if add_offsets:
        offsets = np.arange(0, K) * K
        out = out.astype(np.uint8) + offsets
    return np.ascontiguousarray(out)


def maddness_scan(A_enc, nsplits, N, luts):
    """
    A_enc ... [M, D]
    """
    A_enc = A_enc.reshape(A_enc.shape, order="F")
    out_mat = np.zeros((N, A_enc.shape[0]), order="F").astype(np.uint8)
    LIBMITHRAL_STATIC1.mithral_scan_fp32_t(convert_to_cpp_uint8(A_enc),
                                          2**nsplits,
                                          A_enc.shape[0],
                                          convert_to_cpp_uint8(luts),
                                          convert_to_cpp_uint8(out_mat))
    return out_mat
