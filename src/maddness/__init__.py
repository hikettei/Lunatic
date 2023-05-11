
from .encoder import train_encoder
from .cffi_utils import load_libmithral
# Load CFFI

# dylib object.
LIBMITHRAL_STATIC = None
load_libmithral()
