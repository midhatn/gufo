#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_DSA_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_DSA_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>

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
  float* lat;               // [pos][512]
  float* ik;                // [pos][128] normalized indexer keys
  float* gate;              // [pos][128] compressor gate
  float* pooled;            // [pos / 4][128]
  std::uint16_t* lat_half;  // [pos][512] F16 copy of lat for the batched attend
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

// Single-token forms that read the position from device memory (`pos`), so a
// captured graph replays at any position. max_pools bounds the score grid.
int DsaAppendDev(const DsaCache& cache, const float* lat, const float* ik,
                 const float* gate, const float* ape, const int* pos,
                 hipStream_t stream);
int DsaScoresDev(const float* iq, const float* proj, const float* pooled,
                 int max_pools, float* scores, const int* pos,
                 hipStream_t stream);
int DsaSelectDev(const float* scores, int top_pools, int* selected,
                 const int* pos, hipStream_t stream);
int DsaAttendDev(const float* qabs, const float* lat, const int* selected,
                 int top_pools, float scale, float* ctx, const int* pos,
                 hipStream_t stream);

// Batched forms for `tokens` consecutive tokens starting at position pos0.
// Inputs are [token][...]; scores rows are score_stride apart; selections are
// [token][kDsaTopPools]; qabs and ctx are head-major [64][tokens][512].
int DsaAppendBatch(const DsaCache& cache, const float* lat, const float* ik,
                   const float* gate, const float* ape, int pos0, int tokens,
                   hipStream_t stream);
int DsaScoresBatch(const float* iq, const float* proj, const float* pooled,
                   int pos0, int tokens, long long score_stride, float* scores,
                   hipStream_t stream);
int DsaSelectBatch(const float* scores, long long score_stride, int pos0,
                   int tokens, int top_pools, int* selected,
                   hipStream_t stream);
int DsaAttendBatch(const float* qabs, const std::uint16_t* lat_half,
                   const int* selected, int pos0, int tokens, int top_pools,
                   float scale, float* ctx, hipStream_t stream);

// [token][head][width] <-> [head][token][width] (the per-head GEMM layout).
int HeadMajor(const float* in, float* out, int tokens, int heads, int width,
              hipStream_t stream);
int TokenMajor(const float* in, float* out, int tokens, int heads, int width,
               hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_DSA_HPP_
