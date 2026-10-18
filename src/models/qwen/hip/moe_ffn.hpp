#ifndef GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
#define GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Simple, correctness-first GPU Mixture-of-Experts feed-forward for one token.
///
/// Router softmax and top-k run on the device; the selected experts execute as
/// per-expert loops over the existing quantized GEMV and fused SwiGLU kernels.
/// The routed sum and the gated shared expert are accumulated into `out`. This
/// is the milestone-1 path; the vectorized MoE kernels replace it later.
void MoeFfnForwardSingleToken(const models::QwenLayerWeights& layer,
                              const core::ModelConfig& config, const float* x,
                              float* out, hipStream_t stream);

/// MoE feed-forward for one token with no host involvement: device top-k, the
/// quantized expert vector kernels addressed by the device ids, a gated shared
/// expert and one combine pass. Safe to capture in a HIP graph.
void MoeFfnDecodeToken(const models::QwenLayerWeights& layer,
                       const core::ModelConfig& config,
                       const QwenMoeScratch& scratch, const float* x,
                       float* out, hipStream_t stream);

/// Runs every layer's expert kernels once outside graph capture so the matrix
/// library's workspace pool is sized before the decode graph is captured.
void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
