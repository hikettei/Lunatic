
import os
from cffi import FFI

def load_libmithral():
    global LIBMITHRAL_STATIC
    ffi = FFI()
    cwd = os.getcwd()
    lib_path = f"{cwd}/third_parties/cl-xMatrix/source/kernel/libMithral"
    
    LIBMITHRAL_STATIC = ffi.dlopen(f"{lib_path}.dylib")

    return True

