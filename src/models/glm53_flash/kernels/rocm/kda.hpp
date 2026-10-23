#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_KDA_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_KDA_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

inline constexpr int kKdaHeads = 64;
inline constexpr int kKdaHeadDim = 128;
inline constexpr int kKdaInner = kKdaHeads * kKdaHeadDim;
inline constexpr int kKdaConvTaps = 4;

// Decode-time Kimi Delta Attention pieces for one token. Layouts:
//   qkv     [3][8192]  q, k, v projections (in place -> SiLU(conv))
//   hist    [3][3][8192] last three raw inputs per stream, oldest first
//   state   [64][128][128] delta state S[head][value row][key dim]
// Conv weights are the ggml [4,1,8192] tensors (conv_w[ch * 4 + tap]).

int KdaConv(float* qkv, float* hist, const float* conv_q, const float* conv_k,
            const float* conv_v, hipStream_t stream);

// decay[i] = exp(-5 / (1 + exp((f[i] + dt[i]) * ssm_a[head]))), the per key
// dimension log-decay gate; beta = sigmoid(braw).
int KdaGates(const float* f, const float* dt, const float* ssm_a,
             const float* braw, float* decay, float* beta, hipStream_t stream);

// L2-normalizes q and k per head, applies the delta-rule update and writes
// o = S q / sqrt(128) to o [8192].
int KdaRecur(const float* qkv, const float* decay, const float* beta,
             float* state, float* o, hipStream_t stream);

// In place per head: o = RMSNorm(o) * norm_w * sigmoid(g).
int KdaOut(float* o, const float* norm_w, const float* g, hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_KDA_HPP_
