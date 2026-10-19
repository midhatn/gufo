#if defined(ENGINE_ENABLE_HIP)
#include <cstddef>
#include <cstdlib>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/prefill_fp16.hpp"

namespace gufo::hip {
namespace {
// Below this many rows the per-expert buckets are too thin for grouped GEMMs.
constexpr std::size_t kMinGroupedRows = 64;
}  // namespace

bool QwenGpuExecutor::MoePrefillFfn(const models::QwenLayerWeights& layer,
                                    std::uint32_t layer_index,
                                    std::size_t batch_size) {
  static const bool disabled = [] {
    const char* v = std::getenv("GUFO_QWEN_MOE_PREFILL");
    return v != nullptr && v[0] == '0';
  }();
  if (disabled) {
    return false;
  }
  const auto& config = weights_.config;
  QwenMoePrefill* const backend = model_->MoePrefill(arena_.GetMaxBatch());
  if (backend == nullptr || batch_size < kMinGroupedRows ||
      batch_size > backend->MaxTokens() || !backend->Supports(layer_index)) {
    return false;
  }
  const std::size_t hidden = config.hidden_size;
  const std::size_t shared_ff = config.shared_expert_ff;
  auto& scratch = arena_.moe_prefill;
  hipStream_t stream = arena_.stream;
  // FFN RMSNorm left the FP32 rows in d_normed and their BF16 copy in
  // d_scratch_bf16.
  LaunchFloatToFp16(arena_.d_normed, scratch.x_half, batch_size * hidden,
                    stream);
  const std::uint16_t* split_weights = model_->RouterSplitWeights(layer_index);
  if (split_weights != nullptr && arena_.hipblaslt_gemm != nullptr) {
    LaunchRouterSplitActivations(arena_.d_normed, scratch.router_x, batch_size,
                                 hidden, stream);
    if (!arena_.hipblaslt_gemm->RunBf16(split_weights, scratch.router_x,
                                        scratch.router_y, batch_size,
                                        kRouterSplitRows, 3 * hidden, stream)) {
      split_weights = nullptr;
    } else {
      LaunchRouterSplitLogits(scratch.router_y, scratch.router_logits,
                              scratch.shared_logit, batch_size,
                              config.num_experts, kRouterSplitRows, stream);
    }
  } else {
    split_weights = nullptr;
  }
  if (split_weights == nullptr) {
    LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_gate_inp.data, false,
                      arena_.d_normed, scratch.router_logits, batch_size,
                      config.num_experts, hidden, arena_.d_scratch_bf16,
                      stream);
    LaunchHipblasGEMM(arena_.hipblas_handle, layer.ffn_gate_inp_shexp.data,
                      false, arena_.d_normed, scratch.shared_logit, batch_size,
                      1, hidden, arena_.d_scratch_bf16, stream);
  }
  // Dense shared expert: gate/up projections, SwiGLU, down projection.
  LaunchBatchedQuantGEMM(layer.ffn_gate_shexp.type, layer.ffn_gate_shexp.data,
                         arena_.d_scratch_bf16, scratch.shared_gate, batch_size,
                         shared_ff, hidden, stream);
  LaunchBatchedQuantGEMM(layer.ffn_up_shexp.type, layer.ffn_up_shexp.data,
                         arena_.d_scratch_bf16, scratch.shared_up, batch_size,
                         shared_ff, hidden, stream);
  LaunchBatchedSwiGLUActivation(scratch.shared_gate, scratch.shared_up,
                                scratch.shared_act, arena_.d_scratch_bf16,
                                batch_size * shared_ff, stream);
  LaunchBatchedQuantGEMM(layer.ffn_down_shexp.type, layer.ffn_down_shexp.data,
                         arena_.d_scratch_bf16, scratch.shared_out, batch_size,
                         hidden, shared_ff, stream);
  return backend->Run(layer_index, static_cast<std::uint32_t>(batch_size),
                      scratch.router_logits, scratch.x_half, scratch.shared_out,
                      scratch.shared_logit, arena_.d_ffn_out, stream);
}

}  // namespace gufo::hip
#endif
