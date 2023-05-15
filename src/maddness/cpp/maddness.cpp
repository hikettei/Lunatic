
/*
  Inspired from: https://github.com/dblalock/bolt
*/

#include "maddness.hpp"

#ifndef MAX
#define MAX(x, y) ((x) < (y) ? (y) : (x))
#endif
#ifndef MIN
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#endif

static_assert(__AVX2__, "AVX 2 is required! Try --march=native or -mavx2");

inline __m256 fma(__m256 a, __m256 b, __m256 c) {
  __m256 res = c;
  __asm__("vfmadd231ps %[a], %[b], %[c]"
          : [c] "+x"(res)
          : [a] "x"(a), [b] "x"(b));
  return res;
}

inline __m256i avg_epu8(__m256i a, __m256i b) {
  __m256i res = _mm256_undefined_si256();
  __asm__("vpavgb %[a], %[b], %[c]" : [c] "=x"(res) : [a] "x"(a), [b] "x"(b));
  return res;
}


// based on https://stackoverflow.com/a/51779212/1153180
template <bool Signed = true, bool SameOrder = true>
static inline __m256i pack_ps_epi8_or_epu8(const __m256 &x0, const __m256 &x1,
                                           const __m256 &x2, const __m256 &x3) {
  __m256i a = _mm256_cvtps_epi32(x0);
  __m256i b = _mm256_cvtps_epi32(x1);
  __m256i c = _mm256_cvtps_epi32(x2);
  __m256i d = _mm256_cvtps_epi32(x3);
  __m256i ab = _mm256_packs_epi32(a, b);
  __m256i cd = _mm256_packs_epi32(c, d);
  __m256i abcd = _mm256_undefined_si256();
  if (Signed) {
    abcd = _mm256_packs_epi16(ab, cd);
  } else {
    abcd = _mm256_packus_epi16(ab, cd);
  }
  // packed to one vector, but in [ a_lo, b_lo, c_lo, d_lo | a_hi, b_hi, c_hi,
  // d_hi ] order if you can deal with that in-memory format (e.g. for later
  // in-lane unpack), great, you're done

  if (!SameOrder) {
    return abcd;
  }

  // but if you need sequential order, then vpermd:
  __m256i lanefix = _mm256_permutevar8x32_epi32(abcd, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  return lanefix;
}


template <bool Signed = true, bool SameOrder = true>
static inline __m256i load_4xf32_as_32xepi8_or_epu8(const float *x,
                                                    const __m256 &scales,
                                                    const __m256 &offsets) {
  auto x0 = fma(_mm256_loadu_ps(x), scales, offsets);
  auto x1 = fma(_mm256_loadu_ps(x + 8), scales, offsets);
  auto x2 = fma(_mm256_loadu_ps(x + 16), scales, offsets);
  auto x3 = fma(_mm256_loadu_ps(x + 24), scales, offsets);
  return pack_ps_epi8_or_epu8<Signed, SameOrder>(x0, x1, x2, x3);
}


template <class T> static inline __m256i load_si256i(T *ptr) {
  return _mm256_load_si256((__m256i *)ptr);
}

template <class T> static inline __m128i load_si128i(T *ptr) {
  return _mm_load_si128((__m128i *)ptr);
}


static void maddness_encode_fp32_t(const float *X,
				   int C,
				   int nsplits,
				   int nrows,
				   int ncols,
				   const uint32_t *splitdims,
				   const int8_t *splitvals,
				   const float *scales,
				   const float *offsets,
				   uint8_t* out) {

  /*
    Inputs:
    X* ... FP32 Matrix to be encoded. it is given as **ROW_MAJOR** order with the shape of [nrows, ncols].
    C  ... the number of codebooks.
    splitdims, splitvals, scales, offsets
    -> As it is, but the format is that:
    [ProtoType0(Bucket_0), ProtoType0(Bucket_1), ProtoType0(Bucket_2), ..., ProtoType1()]

    *out ... uint8 matrix [N, C] to store the encoded X*.

    Memo:
    X* ...
    AABBCC
    AABBCC
    AABBCC
    AABBCC
    AABBCC
    
    AA...Proto1, BB...Proto2, CC...Proto3

    => becames
    out ...
    ABC
    ABC
    ABC
    ABC
    ABC
    (Choose the best A,B,C(Encoded) from AA, BB, CC)

    Add offsets to X*, out*, for this function to change the visible araa.

    splitdims' stride(x) = 2**x
  */

  static constexpr int block_nrows = 32; // One Instruction = 32 elements
  const int64_t nblocks = ceil(nrows / (double)block_nrows);
  const int64_t total_buckets_per_tree = 2^nsplits;
  
  const float *x_ptrs[total_buckets_per_tree];
  __m256i current_vsplitval_luts[total_buckets_per_tree];
  __m256  current_vscales[total_buckets_per_tree];
  __m256  current_voffsets[total_buckets_per_tree];

  assert(nrows % block_nrows == 0); // TODO: remove it
  
  int mtree_offset = 0;
  int split_idx = 0;
  
  int STEP = ncols/C;
  
  // X.stride = [1, nrow]
  for (int cth=0;cth<C;cth++) {
    // Processing prototypes-wise: AA -> BB -> CC.
    auto out_ptr = out + (nrows * cth); // out_ptr = out[:, C:]
    mtree_offset = cth * total_buckets_per_tree;
    
    for (int bucket_n=0;bucket_n<total_buckets_per_tree;bucket_n++) {
      auto splitdim = splitdims[mtree_offset + bucket_n];
      splitdim += cth * STEP; // Add Offsets. {AABBCC}[n], n=cth*STEP + splitdim.
      
      x_ptrs[bucket_n] = X + (nrows * splitdim); // x_ptrs[bucket_b] = X[:, splitdim].

      auto splitval_ptr = splitvals + mtree_offset; // First splitval at each level.
      // Here, splitval_ptr = {Bucket(0, 0)}, {Bucket(0, 1), Bucket(1, 1)}, {Bucket(1, 1), Bucket(0, 2), ...}
      
      current_vsplitval_luts[bucket_n] = _mm256_broadcastsi128_si256(load_si128i((const __m128i *)splitval_ptr));
      current_vscales[bucket_n]  = _mm256_set1_ps(scales[mtree_offset + bucket_n]);
      current_voffsets[bucket_n] = _mm256_set1_ps(offsets[mtree_offset + bucket_n]); 
    }

    /*
      __m256 variables: current_XXX is now:

                          / length=nsplits \
      [Bucket(0, 0), Bucket(0, 1), Bucket(0, 2), Bucket(0, 3)] where Bucket(i, t) is bucket, t=tree_level, i=bucket_idx.
    */

    for (int b=0;b<nblocks;b++) {
      __m256i group_idxs = _mm256_setzero_si256();
#pragma unroll
      // tlevel=bucket_n
      for (int bucket_n=0;bucket_n<nsplits;bucket_n++) {
	auto vscales   = current_vscales[bucket_n];
	auto voffsets  = current_voffsets[bucket_n];
	auto vsplitvals_lut = current_vsplitval_luts[bucket_n];
        auto vsplitvals =  _mm256_shuffle_epi8(vsplitvals_lut, group_idxs);

	auto x_ptr = x_ptrs[bucket_n];
	x_ptrs[bucket_n] += block_nrows;
	
	// cmp
	auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true, true>(x_ptr, vscales, voffsets);	
        auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
	auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);
	
	if (bucket_n>0) {
	  // MaddnessHash, 2*idx
	  group_idxs = _mm256_add_epi8(group_idxs, group_idxs);
	}
	
        group_idxs = _mm256_or_si256(group_idxs, masks_0_or_1);
      }
      
      group_idxs = _mm256_permutevar8x32_epi32(group_idxs, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
      _mm256_storeu_si256((__m256i *)out_ptr, group_idxs);
      out_ptr += block_nrows;
    }
    // TO ADD: Reminder
  }
}

// Inspired from: https://github.com/dblalock/bolt
static constexpr bool is_power_of_2(int64_t x) {
  return (x & (x - 1)) == 0 && x > 0;
}

template <int NBytes, int UpcastEvery = 16, int _OutTileSz = 1,
          bool Force16BitOutput = false>
void mithral_scan(const uint8_t *codes, int64_t nblocks, const uint8_t *luts,
                  uint8_t *dists_out) {
  static_assert(NBytes > 0, "Code length <= 0 is not valid");
  static_assert(UpcastEvery % 2 == 0, "UpcastEvery must be even");
  static_assert(UpcastEvery >= 2, "UpcastEvery must be >= 2");
  static_assert(UpcastEvery <= 256, "UpcastEvery must be <= 256");
  static_assert(is_power_of_2(UpcastEvery),
                "UpcastEvery must be a power of 2!");
  static constexpr int ncodebooks = 2 * NBytes;
  static constexpr int ncols = NBytes;
  static constexpr int actually_upcast_every = MIN(UpcastEvery, ncodebooks);
  static constexpr int colgroup_sz = actually_upcast_every / 2;
  static_assert(is_power_of_2(colgroup_sz),
                "Invalid number of columns to unroll at once");
  static constexpr int ncolgroups = ncols / colgroup_sz;
  static_assert(colgroup_sz <= ncodebooks, "WTF, did some math wrong");
  static_assert(ncols % colgroup_sz == 0,
                "Size of column group must evenly number of columns");
  static constexpr bool use_uint8_output = ncolgroups == 1 && !Force16BitOutput;
  static constexpr int OutTileSz = _OutTileSz > 0 ? _OutTileSz : 1;

  int64_t out_stride = use_uint8_output ? nblocks * 32 : 2 * nblocks * 32;
  int lut_stride = ncodebooks * 16;

  uint8_t *out_ptrs[OutTileSz];
  for (int mm = 0; mm < OutTileSz; mm++) {
    out_ptrs[mm] = dists_out + (mm * out_stride);
  }

  // unpack 16B luts into 32B registers
  __m256i lut_arrays[ncodebooks][OutTileSz];
  for (int mm = 0; mm < OutTileSz; mm++) {
    auto lut_ptr = luts + (mm * lut_stride);
    for (uint8_t j = 0; j < NBytes; j++) {
      auto both_luts = load_si256i(lut_ptr);
      lut_ptr += 32;
      auto lut0 = _mm256_permute2x128_si256(both_luts, both_luts, 0 + (0 << 4));
      auto lut1 = _mm256_permute2x128_si256(both_luts, both_luts, 1 + (1 << 4));
      lut_arrays[2 * j][mm] = lut0;
      lut_arrays[2 * j + 1][mm] = lut1;
    }
  }

  for (int64_t i = 0; i < nblocks; i++) {
    // used if ncolgroups > 1, in which case we have to upcast
    __m256i totals_0_15[OutTileSz];
    __m256i totals_16_31[OutTileSz];
    for (int mm = 0; mm < OutTileSz; mm++) {
      totals_0_15[mm] = _mm256_setzero_si256();
      totals_16_31[mm] = _mm256_setzero_si256();
    }

    auto low_4bits_mask = _mm256_set1_epi8(0x0F); // not static so sits in reg

    for (int g = 0; g < ncolgroups; g++) {

      __m256i avg_prev1[OutTileSz];
      __m256i avg_prev2[OutTileSz];
      __m256i avg_prev4[OutTileSz];
      __m256i avg_prev8[OutTileSz];
      __m256i avg_prev16[OutTileSz];
      __m256i avg_prev32[OutTileSz];
      __m256i avg_prev64[OutTileSz];
      __m256i avg_prev128[OutTileSz];
      for (int mm = 0; mm < OutTileSz; mm++) {
        avg_prev1[mm] = _mm256_undefined_si256();
        avg_prev2[mm] = _mm256_undefined_si256();
        avg_prev4[mm] = _mm256_undefined_si256();
        avg_prev8[mm] = _mm256_undefined_si256();
        avg_prev16[mm] = _mm256_undefined_si256();
        avg_prev32[mm] = _mm256_undefined_si256();
        avg_prev64[mm] = _mm256_undefined_si256();
        avg_prev128[mm] = _mm256_undefined_si256();
      }

#pragma unroll
      for (int gg = 0; gg < colgroup_sz; gg++) {
        auto j = (g * colgroup_sz) + gg;

        auto x_col = load_si256i(codes);
        codes += 32;
        auto x_low = _mm256_and_si256(x_col, low_4bits_mask);
        auto x_shft = _mm256_srli_epi16(x_col, 4);
        auto x_high = _mm256_and_si256(x_shft, low_4bits_mask);

        for (int mm = 0; mm < OutTileSz; mm++) {
          auto lut_low = lut_arrays[2 * j][mm];
          auto lut_high = lut_arrays[2 * j + 1][mm];

          auto dists_low = _mm256_shuffle_epi8(lut_low, x_low);
          auto dists_high = _mm256_shuffle_epi8(lut_high, x_high);

          auto avgs = _mm256_avg_epu8(dists_low, dists_high);

          // update running averages; this is messy because you
          // need the current and previous average to be over the same
          // number of values, or else it's a weird weighted average
          // instead of a true average;
          // note that we need to use inline asm to get the right
          // instruction here on my machine for unclear reasons
          if (gg % 128 == 127) {
            auto new_avg_prev2 = avg_epu8(avg_prev1[mm], avgs);
            auto new_avg_prev4 = avg_epu8(avg_prev2[mm], new_avg_prev2);
            auto new_avg_prev8 = avg_epu8(avg_prev4[mm], new_avg_prev4);
            auto new_avg_prev16 = avg_epu8(avg_prev8[mm], new_avg_prev8);
            auto new_avg_prev32 = avg_epu8(avg_prev16[mm], new_avg_prev16);
            auto new_avg_prev64 = avg_epu8(avg_prev32[mm], new_avg_prev32);
            avg_prev128[mm] = avg_epu8(avg_prev64[mm], new_avg_prev64);
          }
          if (gg % 64 == 63) {
            auto new_avg_prev2 = avg_epu8(avg_prev1[mm], avgs);
            auto new_avg_prev4 = avg_epu8(avg_prev2[mm], new_avg_prev2);
            auto new_avg_prev8 = avg_epu8(avg_prev4[mm], new_avg_prev4);
            auto new_avg_prev16 = avg_epu8(avg_prev8[mm], new_avg_prev8);
            auto new_avg_prev32 = avg_epu8(avg_prev16[mm], new_avg_prev16);
            avg_prev64[mm] = avg_epu8(avg_prev32[mm], new_avg_prev32);
          }
          if (gg % 32 == 31) {
            auto new_avg_prev2 = avg_epu8(avg_prev1[mm], avgs);
            auto new_avg_prev4 = avg_epu8(avg_prev2[mm], new_avg_prev2);
            auto new_avg_prev8 = avg_epu8(avg_prev4[mm], new_avg_prev4);
            auto new_avg_prev16 = avg_epu8(avg_prev8[mm], new_avg_prev8);
            avg_prev32[mm] = avg_epu8(avg_prev16[mm], new_avg_prev16);
          }
          if (gg % 16 == 15) {
            auto new_avg_prev2 = avg_epu8(avg_prev1[mm], avgs);
            auto new_avg_prev4 = avg_epu8(avg_prev2[mm], new_avg_prev2);
            auto new_avg_prev8 = avg_epu8(avg_prev4[mm], new_avg_prev4);
            avg_prev16[mm] = avg_epu8(avg_prev8[mm], new_avg_prev8);
          }
          if (gg % 8 == 7) {
            auto new_avg_prev2 = avg_epu8(avg_prev1[mm], avgs);
            auto new_avg_prev4 = avg_epu8(avg_prev2[mm], new_avg_prev2);
            avg_prev8[mm] = avg_epu8(avg_prev4[mm], new_avg_prev4);
          }
          if (gg % 4 == 3) {
            auto new_avg_prev2 = avg_epu8(avg_prev1[mm], avgs);
            avg_prev4[mm] = avg_epu8(avg_prev2[mm], new_avg_prev2);
          }
          if (gg % 2 == 1) {
            avg_prev2[mm] = avg_epu8(avg_prev1[mm], avgs);
          } else {
            avg_prev1[mm] = avgs;
          }
        }
      }

      for (int mm = 0; mm < OutTileSz; mm++) {
        auto group_avg = colgroup_sz == 1    ? avg_prev1[mm]
                         : colgroup_sz == 2  ? avg_prev2[mm]
                         : colgroup_sz == 4  ? avg_prev4[mm]
                         : colgroup_sz == 8  ? avg_prev8[mm]
                         : colgroup_sz == 16 ? avg_prev16[mm]
                         : colgroup_sz == 32 ? avg_prev32[mm]
                         : colgroup_sz == 64 ? avg_prev64[mm]
                                             : avg_prev128[mm];
        if (use_uint8_output) { // write out 8b values
          _mm256_stream_si256((__m256i *)out_ptrs[mm], group_avg);
          out_ptrs[mm] += 32;
        } else {
          auto avgs_0_15 =
              _mm256_cvtepi8_epi16(_mm256_extracti128_si256(group_avg, 0));
          auto avgs_16_31 =
              _mm256_cvtepi8_epi16(_mm256_extracti128_si256(group_avg, 1));
          totals_0_15[mm] = _mm256_add_epi16(totals_0_15[mm], avgs_0_15);
          totals_16_31[mm] = _mm256_add_epi16(totals_16_31[mm], avgs_16_31);
        }
      }
    }
    if (!use_uint8_output) {
      for (int mm = 0; mm < OutTileSz; mm++) {
        _mm256_stream_si256((__m256i *)(out_ptrs[mm] + 0), totals_0_15[mm]);
        _mm256_stream_si256((__m256i *)(out_ptrs[mm] + 32), totals_16_31[mm]);
        out_ptrs[mm] += 64;
      }
    }
  }
}

template <int UpcastEvery = 64>
void mithral_scan_notile(const uint8_t *codes, int64_t nblocks, int ncodebooks,
                         const uint8_t *luts, uint8_t *out) {
  switch (ncodebooks) {
  case 2:
    mithral_scan_notile<1, UpcastEvery>(codes, nblocks, luts, out);
    break;
  case 4:
    mithral_scan_notile<2, UpcastEvery>(codes, nblocks, luts, out);
    break;
  case 8:
    mithral_scan_notile<4, UpcastEvery>(codes, nblocks, luts, out);
    break;
  case 16:
    mithral_scan_notile<8, UpcastEvery>(codes, nblocks, luts, out);
    break;
  case 32:
    mithral_scan_notile<16, UpcastEvery>(codes, nblocks, luts, out);
    break;
  case 64:
    mithral_scan_notile<32, UpcastEvery>(codes, nblocks, luts, out);
    break;
  case 128:
    mithral_scan_notile<64, UpcastEvery>(codes, nblocks, luts, out);
    break;
  default:
    assert(false); // unsupported ncodebooks
  }
}

template <int UpcastEvery = 64, int OutTileSz = 1>
void mithral_scan(const uint8_t *codes, int64_t nblocks, int ncodebooks,
                  const uint8_t *luts, uint8_t *out) {
  switch (ncodebooks) {
  case 2:
    mithral_scan<1, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  case 4:
    mithral_scan<2, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  case 8:
    mithral_scan<4, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  case 16:
    mithral_scan<8, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  case 32:
    mithral_scan<16, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  case 64:
    mithral_scan<32, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  case 128:
    mithral_scan<64, UpcastEvery, OutTileSz>(codes, nblocks, luts, out);
    break;
  default:
    assert(false); // unsupported ncodebooks
  }
}

template <int UpcastEvery = 64>
void mithral_scan_nochunk(const uint8_t *codes, int64_t nblocks, int ncodebooks,
                          int noutputs, const uint8_t *luts,
                          uint8_t *dists_out) {
  static constexpr int block_nrows = 32;
  static constexpr int lut_sz = 16;
  auto out_ptr = dists_out;
  auto out_stride = nblocks * block_nrows;
  auto lut_ptr = luts;
  auto lut_stride = ncodebooks * lut_sz;

  for (int i = 0; i < noutputs; i++) {
    mithral_scan_notile<UpcastEvery>(codes, nblocks, ncodebooks, lut_ptr,
                                     out_ptr);
    out_ptr += out_stride;
    lut_ptr += lut_stride;
  }
}

template <int UpcastEvery = 128, int _OutTileSz = 2>
void mithral_scan(const uint8_t *codes, int64_t nblocks, int ncodebooks,
                  int noutputs, const uint8_t *luts, uint8_t *dists_out) {
  static constexpr int OutTileSz = _OutTileSz > 0 ? _OutTileSz : 1;
  static constexpr int block_nrows = 32;
  static constexpr int lut_sz = 16;
  // static constexpr int chunk_nrows = 999999;  // no chunking
  // static constexpr int chunk_nblocks = chunk_nrows / block_nrows;

  static constexpr int target_chunk_nbytes = 24 * 1024; // most of L1 cache
  int codes_row_nbytes = ncodebooks / 2;
  int codes_block_nbytes = codes_row_nbytes * block_nrows;
  int chunk_nblocks = target_chunk_nbytes / codes_block_nbytes;
  int chunk_nrows = chunk_nblocks * block_nrows;

  auto codes_row_stride = ncodebooks / 2;
  auto codes_chunk_stride = codes_row_stride * chunk_nrows;
  auto out_chunk_stride = chunk_nrows;
  auto out_col_stride = nblocks * block_nrows;
  auto lut_chunk_stride = 0;
  auto lut_col_stride = ncodebooks * lut_sz;

  auto nchunks = (nblocks + chunk_nblocks - 1) / chunk_nblocks;
  for (int chunk = 0; chunk < nchunks;
       chunk++) { // for each chunk of input rows
    int64_t use_nblocks = chunk_nblocks;
    if (chunk == (nchunks - 1)) { // handle last chunk
      auto nblocks_done = chunk * chunk_nblocks;
      use_nblocks = nblocks - nblocks_done;
    }
    auto codes_ptr = codes + (chunk * codes_chunk_stride);
    auto out_ptr = dists_out + (chunk * out_chunk_stride);
    auto lut_ptr = luts + (chunk * lut_chunk_stride);

    int nfullgroups_out = noutputs / OutTileSz;
    for (int g = 0; g < nfullgroups_out; g++) {
      mithral_scan<UpcastEvery, OutTileSz>(codes_ptr, use_nblocks, ncodebooks,
                                           lut_ptr, out_ptr);
      out_ptr += out_col_stride * OutTileSz;
      lut_ptr += lut_col_stride * OutTileSz;
    }
    int ntrailing_outputs = noutputs % OutTileSz;
    for (int m = 0; m < ntrailing_outputs; m++) {
      mithral_scan<UpcastEvery, 1>(codes_ptr, use_nblocks, ncodebooks, lut_ptr,
                                   out_ptr);
      out_ptr += out_col_stride * OutTileSz;
      lut_ptr += lut_col_stride * OutTileSz;
    }
  }
}






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
		       uint8_t* out) {
    maddness_encode_fp32_t(X, C, nsplits, nrows, ncols, splitdims, splitvals, scales, offsets, out);
  }

  void maddness_scan(const uint8_t* encoded_mat,
		     int C,
		     int M,
		     const uint8_t* luts,
		     uint8_t* out_mat) {
    // luts, out_mat ... RowMajor
    // encoded ... column major
    mithral_scan<16,2>(encoded_mat, 32, C, M, luts, out_mat);
  };
}
