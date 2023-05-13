
#include "immintrin.h"
#include <assert.h>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <sys/types.h>
#include <type_traits>

extern "C" {
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
}
