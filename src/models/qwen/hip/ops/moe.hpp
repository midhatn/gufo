#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Softmax over `num_experts` router logits, selection of the `num_used`
/// highest-probability experts (ties resolved toward the lower index), and
/// renormalization of the selected weights to sum one. `shared_logit` is the
/// single scalar router score for the shared expert; its sigmoid is written to
/// `shared_weight[0]`. Runs as one block so the whole vocabulary of experts
/// stays in shared memory.
void LaunchMoeRouterTopK(const float* router_logits, const float* shared_logit,
                         std::int32_t* expert_ids, float* expert_weights,
                         float* shared_weight, std::uint32_t num_experts,
                         std::uint32_t num_used, hipStream_t stream = nullptr);

/// Elementwise weighted accumulation: out += scale * add.
void LaunchMoeScaledAccumulate(float* out, const float* add, float scale,
                               std::size_t dim, hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
