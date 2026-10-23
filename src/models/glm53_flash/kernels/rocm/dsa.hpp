#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_DSA_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_DSA_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

inline constexpr int kDsaHeads = 64;
inline constexpr int kDsaLatent = 512;
inline constexpr int kDsaIndexHeads = 32;
inline constexpr int kDsaIndexDim = 128;
inline constexpr int kDsaPool = 4;
inline constexpr int kDsaTopPools = 512;
// Visible positions: selected pools of kDsaPool plus up to kDsaPool - 1 tail.
inline constexpr int kDsaMaxVisible = kDsaTopPools * kDsaPool + kDsaPool - 1;

// In-place single-block norms for n <= 16384 elements.
// `rows` consecutive rows of n elements are normalized independently.
int RmsNorm(float* x, const float* w, int n, float eps, hipStream_t stream,
            int rows = 1);
int LayerNorm(float* x, const float* w, const float* b, int n, float eps,
              hipStream_t stream, int rows = 1);

// Per-layer device caches, one row per token (pooled: one per complete pool).
struct DsaCache {
  float* lat;     // [pos][512]
  float* ik;      // [pos][128] normalized indexer keys
  float* gate;    // [pos][128] compressor gate
  float* pooled;  // [pos / 4][128]
};

// Stores token `pos` and, when it completes a pool, writes the pooled key.
// ape is the [4][128] compressor positional bias (ape[d + 128 * slot]).
int DsaAppend(const DsaCache& cache, const float* lat, const float* ik,
              const float* gate, const float* ape, int pos, hipStream_t stream);

// scores[p] = sum_h relu(iq_h . pooled_p) * proj[h] / sqrt(128 * 32).
int DsaScores(const float* iq, const float* proj, const float* pooled,
              int npool, float* scores, hipStream_t stream);

// Picks min(npool, top_pools) highest-scoring pools, ties to the lower pool
// index, and writes them in ascending pool order. scratch: 3 * npool uint32.
int DsaSelect(const float* scores, int npool, int top_pools, int* selected,
              hipStream_t stream);

// Absorbed-MLA attention over the selected pools plus the unpooled tail of
// token `pos`. qabs and ctx are [64][512]; scale multiplies the scores.
int DsaAttend(const float* qabs, const float* lat, const int* selected,
              int nselected, int pos, float scale, float* ctx,
              hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_DSA_HPP_
