#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MHC_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MHC_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

inline constexpr int kHcStreams = 4;
inline constexpr int kHcHidden = 4096;
inline constexpr int kHcMixes = 24;

// hyper-connection pre-mix for one token. x is [4][4096] residual streams.
// fn is [24][16384] F32, scale [3], base [24]. Writes the block input
// y = RMSNorm(sum_s x_s * pre[s]) * norm_w, post[4] and comb[16]
// (comb[dst + 4 * src]); all device pointers. `tokens` consecutive tokens are
// processed independently, one thread block each (x, y, post and comb advance
// by one token's worth per index).
int HcPre(const float* x, const float* fn, const float* scale,
          const float* base, const float* norm_w, float hc_eps, float* y,
          float* post, float* comb, hipStream_t stream, int tokens = 1);

// next[d] = block * post[d] + sum_s x[s] * comb[d + 4 * s]. next != x.
int HcPost(const float* x, const float* block, const float* post,
           const float* comb, float* next, hipStream_t stream, int tokens = 1);

// out = RMSNorm(mean_s x_s) * w.
int HeadNorm(const float* x, const float* w, float* out, hipStream_t stream,
             int rows = 1);

// dst[r * dst_stride + tap_offset + i] = mean over the 4 streams of
// x[(first_row + r)][s][i] for r < rows (a drafter feature tap).
int MeanStreams(const float* x, int first_row, int rows, float* dst,
                long long dst_stride, long long tap_offset, hipStream_t stream);

// out[r] = index of the largest logit of row r (lowest index on ties).
int ArgmaxRows(const float* logits, int rows, int vocab, int* out,
               hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MHC_HPP_
