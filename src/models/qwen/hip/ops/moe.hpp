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
                         std::uint32_t num_used, std::uint32_t rows = 1,
                         hipStream_t stream = nullptr);

/// Elementwise weighted accumulation: out += scale * add.
void LaunchMoeScaledAccumulate(float* out, const float* add, float scale,
                               std::size_t dim, hipStream_t stream = nullptr);

/// out[i] = sum_s slot_weight[s] * slot_out[s][i] + shared_weight[0] *
/// shared_out[i]: the routed experts and the gated shared expert in one pass.
void LaunchMoeCombine(float* out, const float* slot_out,
                      const float* slot_weight, const float* shared_out,
                      const float* shared_weight, std::uint32_t used,
                      std::uint32_t hidden, std::uint32_t rows = 1,
                      hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_HPP_
