
import os
from cffi import FFI
import numpy as np

def load_libmithral():
    global LIBMITHRAL_STATIC
    ffi = FFI()
    cwd = os.getcwd()
    lib_path = f"{cwd}/third_parties/cl-xMatrix/source/kernel/libMithral"

    ffi.cdef("""
      void mithral_encode_fp32_t(const float *X,
			     int64_t nrows,
			     int ncols,
			     const uint32_t *splitdims,
			     const int8_t *all_splitvals,
			     const float *scales,
			     const float *offsets,
			     int ncodebooks,
			     uint8_t *out);
    
     void mithral_scan_fp32_t(const uint8_t* encoded_mat,
			   int ncodebooks,
			   int M,
			   const uint8_t* luts,
			   uint8_t* out_mat);
    
    void mithral_lut_fp32_t(const float *Q, int nrows, int ncols, int ncodebooks,
                       const float *centroids, float *out_offset_sum,
                       float *out_scale, float *tmp_lut_f32,
		       uint8_t *out);
    """)
    LIBMITHRAL_STATIC = ffi.dlopen(f"{lib_path}.dylib")
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
def maddness_encode(X, splitdims, splitvals, scales, offsets, ncodebooks, add_offsets=True):
    out = np.zeros((X.shape[0], ncodebooks), dtype=np.int8, order="F")
    LIBMITHRAL_STATIC.mithral_encode_fp32_t(convert_to_cpp_float(X),
                                            X.shape[0],
                                            X.shape[1],
                                            convert_to_cpp_uint32(splitdims),
                                            convert_to_cpp_int8(splitvals),
                                            convert_to_cpp_float(scales),
                                            convert_to_cpp_float(offsets),
                                            ncodebooks,
                                            convert_to_cpp_uint8(out))
    if add_offsets:
        offsets = np.arange(0, ncodebooks) * ncodebooks
        out = out.astype(np.int32) + offsets
    return out

def maddness_lut(B, all_prototypes, C, K):
    ffi = FFI()
    ncols, nrows = B.shape
    out     = np.zeros((C, K, ncols), dtype=np.uint8, order="F")
    tmp_f32 = np.zeros((C, K, ncols), dtype=np.float32, order="F")
    offset_sum = ffi.new("float*", 0.0)
    scale_sum  = ffi.new("float*", 0.0)
    LIBMITHRAL_STATIC.mithral_lut_fp32_t(convert_to_cpp_float(B),
                                         ncols,
                                         nrows,
                                         K,
                                         convert_to_cpp_float(all_prototypes),
                                         offset_sum,
                                         scale_sum,
                                         convert_to_cpp_float(tmp_f32),
                                         convert_to_cpp_uint8(out))
    
    return out, float(scale_sum[0]), float(offset_sum[0])

def maddness_scan(A_enc, nsplits, N, luts):
    """
    A_enc ... [M, D]
    """
    A_enc = A_enc.reshape(A_enc.shape, order="F")
    out_mat = np.zeros((N, A_enc.shape[0]), order="F").astype(np.uint8)
    LIBMITHRAL_STATIC.mithral_scan_fp32_t(convert_to_cpp_uint8(A_enc),
                                          2**nsplits,
                                          A_enc.shape[0],
                                          convert_to_cpp_uint8(luts),
                                          convert_to_cpp_uint8(out_mat))
    return out_mat
