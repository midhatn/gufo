#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_ffn.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/moe_vec.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"

namespace gufo::hip {
namespace {

/// Lazily allocated device and host scratch for the simple MoE path. Sized from
/// the first MoE configuration it sees; Qwen MoE hosted layouts do not change
/// within a process. Replaced by arena-owned scratch in the vectorized path.
struct MoeFfnScratch {
  float* d_router_logits{nullptr};
  float* d_shared_logit{nullptr};
  std::int32_t* d_expert_ids{nullptr};
  float* d_expert_weights{nullptr};
  float* d_shared_weight{nullptr};
  float* d_activation{nullptr};
  float* d_expert_out{nullptr};
  std::vector<std::int32_t> h_expert_ids;
  std::vector<float> h_expert_weights;
  std::vector<float> h_shared_weight;
  std::uint32_t experts{0};
  std::uint32_t used{0};
  std::uint32_t ff{0};
  std::uint32_t sff{0};
  std::uint32_t hidden{0};

  ~MoeFfnScratch() {
    if (d_router_logits != nullptr)
      (void)hipFree(d_router_logits);
    if (d_shared_logit != nullptr)
      (void)hipFree(d_shared_logit);
    if (d_expert_ids != nullptr)
      (void)hipFree(d_expert_ids);
    if (d_expert_weights != nullptr)
      (void)hipFree(d_expert_weights);
    if (d_shared_weight != nullptr)
      (void)hipFree(d_shared_weight);
    if (d_activation != nullptr)
      (void)hipFree(d_activation);
    if (d_expert_out != nullptr)
      (void)hipFree(d_expert_out);
  }

  void Ensure(const core::ModelConfig& config) {
    if (experts == config.num_experts && used == config.num_experts_used &&
        ff == config.expert_ff && sff == config.shared_expert_ff &&
        hidden == config.hidden_size) {
      return;
    }
    if (d_router_logits != nullptr) {
      (void)hipFree(d_router_logits);
      (void)hipFree(d_shared_logit);
      (void)hipFree(d_expert_ids);
      (void)hipFree(d_expert_weights);
      (void)hipFree(d_shared_weight);
      (void)hipFree(d_activation);
      (void)hipFree(d_expert_out);
    }
    experts = config.num_experts;
    used = config.num_experts_used;
    ff = config.expert_ff;
    sff = config.shared_expert_ff;
    hidden = config.hidden_size;
    const std::size_t activation_width = ff > sff ? ff : sff;
    HIP_CHECK(hipMalloc(&d_router_logits, experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_shared_logit, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_ids, used * sizeof(std::int32_t)));
    HIP_CHECK(hipMalloc(&d_expert_weights, used * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_shared_weight, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_activation, activation_width * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_expert_out, hidden * sizeof(float)));
    h_expert_ids.resize(used);
    h_expert_weights.resize(used);
    h_shared_weight.resize(1);
  }
};

MoeFfnScratch& GetMoeFfnScratch() {
  static MoeFfnScratch scratch;
  return scratch;
}

[[nodiscard]] const void* ExpertSlice(const models::QwenTensorRef& stacked,
                                      std::size_t expert, std::size_t rows,
                                      std::size_t cols) noexcept {
  const std::size_t stride =
      ::gufo::quant::QuantizedRowBytes(stacked.type, cols) * rows;
  return static_cast<const std::uint8_t*>(stacked.data) + (expert * stride);
}

}  // namespace

void MoeFfnForwardSingleToken(const models::QwenLayerWeights& layer,
                              const core::ModelConfig& config, const float* x,
                              float* out, hipStream_t stream) {
  const std::size_t experts = config.num_experts;
  const std::size_t used = config.num_experts_used;
  const std::size_t ff = config.expert_ff;
  const std::size_t shared_ff = config.shared_expert_ff;
  const std::size_t hidden = config.hidden_size;

  auto& scratch = GetMoeFfnScratch();
  scratch.Ensure(config);

  LaunchGEMV(layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, x,
             scratch.d_shared_logit, 1, hidden, stream);
  LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, x,
             scratch.d_router_logits, experts, hidden, stream);
  LaunchMoeRouterTopK(scratch.d_router_logits, scratch.d_shared_logit,
                      scratch.d_expert_ids, scratch.d_expert_weights,
                      scratch.d_shared_weight,
                      static_cast<std::uint32_t>(experts),
                      static_cast<std::uint32_t>(used), stream);

  HIP_CHECK(hipMemcpyAsync(scratch.h_expert_ids.data(), scratch.d_expert_ids,
                           used * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                           stream));
  HIP_CHECK(hipMemcpyAsync(scratch.h_expert_weights.data(),
                           scratch.d_expert_weights, used * sizeof(float),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipMemcpyAsync(scratch.h_shared_weight.data(),
                           scratch.d_shared_weight, sizeof(float),
                           hipMemcpyDeviceToHost, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipMemsetAsync(out, 0, hidden * sizeof(float), stream));

  for (std::size_t i = 0; i < used; ++i) {
    const std::size_t e = static_cast<std::size_t>(scratch.h_expert_ids[i]);
    const float weight = scratch.h_expert_weights[i];
    LaunchFusedSwiGLUGEMV(
        ExpertSlice(layer.ffn_gate_exps, e, ff, hidden),
        layer.ffn_gate_exps.type, ExpertSlice(layer.ffn_up_exps, e, ff, hidden),
        layer.ffn_up_exps.type, x, scratch.d_activation, ff, hidden, stream);
    LaunchGEMV(ExpertSlice(layer.ffn_down_exps, e, hidden, ff),
               layer.ffn_down_exps.type, scratch.d_activation,
               scratch.d_expert_out, hidden, ff, stream);
    LaunchMoeScaledAccumulate(out, scratch.d_expert_out, weight, hidden,
                              stream);
  }

  LaunchFusedSwiGLUGEMV(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
                        layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, x,
                        scratch.d_activation, shared_ff, hidden, stream);
  LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
             scratch.d_activation, scratch.d_expert_out, hidden, shared_ff,
             stream);
  LaunchMoeScaledAccumulate(out, scratch.d_expert_out,
                            scratch.h_shared_weight[0], hidden, stream);
}

void MoeFfnDecodeToken(const models::QwenLayerWeights& layer,
                       const core::ModelConfig& config,
                       const QwenMoeScratch& scratch, const float* x,
                       float* out, hipStream_t stream) {
  const int experts = static_cast<int>(config.num_experts);
  const int used = static_cast<int>(config.num_experts_used);
  const int ff = static_cast<int>(config.expert_ff);
  const int hidden = static_cast<int>(config.hidden_size);
  const int shared_ff = static_cast<int>(config.shared_expert_ff);

  LaunchGEMV(layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, x,
             scratch.shared_logit, 1, hidden, stream);
  LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, x,
             scratch.router_logits, experts, hidden, stream);
  LaunchMoeRouterTopK(scratch.router_logits, scratch.shared_logit, scratch.ids,
                      scratch.weights, scratch.shared_weight,
                      static_cast<std::uint32_t>(experts),
                      static_cast<std::uint32_t>(used), stream);
  // activation[s] = SiLU(gate_e x) * (up_e x) for the selected experts e.
  if (qfn_mmq_moe_gated_vec(static_cast<int>(layer.ffn_gate_exps.type),
                            layer.ffn_gate_exps.data, layer.ffn_up_exps.data, x,
                            scratch.ids, scratch.act, ff, hidden, 1, experts,
                            used, stream) != 0) {
    throw std::runtime_error("qwen35moe expert gate/up projection failed");
  }
  // Each slot row multiplies its own expert's down matrix: used rows, one
  // expert each.
  if (qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                      layer.ffn_down_exps.data, scratch.act, scratch.ids,
                      scratch.down_out, hidden, ff, used, experts, 1, stream,
                      nullptr, nullptr) != 0) {
    throw std::runtime_error("qwen35moe expert down projection failed");
  }
  LaunchFusedSwiGLUGEMV(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
                        layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, x,
                        scratch.shared_act, shared_ff, hidden, stream);
  LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
             scratch.shared_act, scratch.shared_out, hidden, shared_ff, stream);
  LaunchMoeCombine(out, scratch.down_out, scratch.weights, scratch.shared_out,
                   scratch.shared_weight, static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(hidden), stream);
}

void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream) {
  const int experts = static_cast<int>(config.num_experts);
  const int used = static_cast<int>(config.num_experts_used);
  const int ff = static_cast<int>(config.expert_ff);
  const int hidden = static_cast<int>(config.hidden_size);
  HIP_CHECK(
      hipMemsetAsync(scratch.ids, 0, used * sizeof(std::int32_t), stream));
  for (const auto& layer : weights.layers) {
    (void)qfn_mmq_moe_gated_vec(
        static_cast<int>(layer.ffn_gate_exps.type), layer.ffn_gate_exps.data,
        layer.ffn_up_exps.data, x, scratch.ids, scratch.act, ff, hidden, 1,
        experts, used, stream);
    (void)qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                          layer.ffn_down_exps.data, scratch.act, scratch.ids,
                          scratch.down_out, hidden, ff, used, experts, 1,
                          stream, nullptr, nullptr);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
