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
// (comb[dst + 4 * src]); all device pointers, one thread block.
int HcPre(const float* x, const float* fn, const float* scale,
          const float* base, const float* norm_w, float hc_eps, float* y,
          float* post, float* comb, hipStream_t stream);

// next[d] = block * post[d] + sum_s x[s] * comb[d + 4 * s]. next != x.
int HcPost(const float* x, const float* block, const float* post,
           const float* comb, float* next, hipStream_t stream);

// out = RMSNorm(mean_s x_s) * w.
int HeadNorm(const float* x, const float* w, float* out, hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MHC_HPP_
