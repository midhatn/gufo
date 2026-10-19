#ifndef GUFO_MODELS_QWEN_HIP_MOE_PREFILL_HPP_
#define GUFO_MODELS_QWEN_HIP_MOE_PREFILL_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace gufo::models {
struct QwenModelWeights;
}

namespace gufo::hip {

/// Routed-expert half of a qwen35moe prefill feed-forward for a batch of
/// tokens: top-k routing, expert-sorted row compaction, grouped quantized
/// GEMMs on the F16 matrix cores, and the combine with the shared expert. The
/// implementation lives with the other MoE GEMM kernels (Flash-Next library)
/// and registers itself at startup; without it prefill keeps the per-token
/// path.
class QwenMoePrefill {
public:
  virtual ~QwenMoePrefill() = default;

  /// Largest batch Run accepts.
  [[nodiscard]] virtual std::uint32_t MaxTokens() const noexcept = 0;
  /// True when `layer` has expert weights the grouped kernels can read.
  [[nodiscard]] virtual bool Supports(std::uint32_t layer) const noexcept = 0;
  /// router_logits [n][experts] F32, x_half [n][hidden], shared_out
  /// [n][hidden] F32 and shared_logit [n] F32 (the shared expert's gate score);
  /// writes out [n][hidden] = sum_s w_s * expert_s(x) + sigmoid(logit) *
  /// shared.
  [[nodiscard]] virtual bool Run(std::uint32_t layer, std::uint32_t n_tokens,
                                 const float* router_logits,
                                 const __half* x_half, const float* shared_out,
                                 const float* shared_logit, float* out,
                                 hipStream_t stream) = 0;
};

using QwenMoePrefillFactory = std::function<std::unique_ptr<QwenMoePrefill>(
    const models::QwenModelWeights&, std::uint32_t, std::string*)>;

void SetQwenMoePrefillFactory(QwenMoePrefillFactory factory);
std::unique_ptr<QwenMoePrefill> CreateQwenMoePrefill(
    const models::QwenModelWeights& weights, std::uint32_t max_tokens,
    std::string* error);

}  // namespace gufo::hip
#endif
#endif  // GUFO_MODELS_QWEN_HIP_MOE_PREFILL_HPP_
