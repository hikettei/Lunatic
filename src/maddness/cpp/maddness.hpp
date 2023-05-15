
#include "immintrin.h"
#include <assert.h>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <sys/types.h>
#include <type_traits>


void mithral_scan(const uint8_t *codes, int64_t nblocks, int ncodebooks,
                  int noutputs, const uint8_t *luts, uint8_t *dists_out);


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

  void maddness_scan(const uint8_t* encoded_mat,
		     int C,
		     int M,
		     const uint8_t* luts,
		     uint8_t* out_mat);
}
