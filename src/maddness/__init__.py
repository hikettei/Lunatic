
from .encoder import train_encoder
from .decoder import (
    construct_lut,
    lut_scan
)

from .cffi_utils import (
    maddness_encode,
    load_libmithral
)

from .hash_function_helper import flatten_buckets

# Load CFFI

# dylib object.
LIBMITHRAL_STATIC = None
load_libmithral()
