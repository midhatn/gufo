#ifndef GUFO_MODELS_QWEN_HIP_NPU_FFN_HPP_
#define GUFO_MODELS_QWEN_HIP_NPU_FFN_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace gufo::models {
struct QwenModelWeights;
}

namespace gufo::hip {

/// Runs the leading output columns of a layer's FFN gate/up projections (and
/// their SwiGLU) on the NPU while the GPU computes the rest. Prefill chunks of
/// exactly `Tokens()` rows only; the GPU path handles everything else.
class QwenNpuFfn {
public:
  virtual ~QwenNpuFfn() = default;

  [[nodiscard]] virtual std::uint32_t Tokens() const noexcept = 0;
  /// Leading FFN columns the NPU computes for `layer`; 0 when the layer stays
  /// on the GPU.
  [[nodiscard]] virtual std::uint32_t Columns(
      std::uint32_t layer) const noexcept = 0;
  /// Encodes the normed FP16 activations (Tokens() x hidden) and enqueues the
  /// NPU runs behind the stream's earlier work. Returns false, having queued
  /// nothing, when the NPU is busy or unhealthy; the caller then computes every
  /// column on the GPU.
  [[nodiscard]] virtual bool Begin(std::uint32_t layer, const void* normed,
                                   hipStream_t stream) = 0;
  /// Blocks the stream until the NPU runs finish and writes the SwiGLU result
  /// into columns [0, Columns(layer)) of `out` (FP16, row pitch `ld`).
  virtual void Finish(std::uint32_t layer, void* out, std::size_t ld,
                      hipStream_t stream) = 0;

  /// Leading output columns of the linear-attention `attn_qkv` and `attn_gate`
  /// projections the NPU computes for `layer`; 0 when they stay on the GPU.
  [[nodiscard]] virtual std::uint32_t QkvColumns(
      std::uint32_t layer) const noexcept = 0;
  [[nodiscard]] virtual std::uint32_t GateColumns(
      std::uint32_t layer) const noexcept = 0;
  /// As Begin for those two projections of the normed FP16 activations.
  [[nodiscard]] virtual bool BeginLinear(std::uint32_t layer,
                                         const void* normed,
                                         hipStream_t stream) = 0;
  /// Waits for the NPU and decodes the leading columns into the FP32 outputs
  /// (row pitches `qkv_ld` and `gate_ld`).
  virtual void FinishLinear(std::uint32_t layer, float* qkv, std::size_t qkv_ld,
                            float* gate, std::size_t gate_ld,
                            hipStream_t stream) = 0;
};

using QwenNpuFfnFactory = std::function<std::unique_ptr<QwenNpuFfn>(
    const models::QwenModelWeights&, std::string*)>;

/// The NPU runtime registers itself here at startup; without a registered
/// factory (or without GUFO_NPU_DIR) the 27B runs on the GPU alone.
void SetQwenNpuFfnFactory(QwenNpuFfnFactory factory);
std::unique_ptr<QwenNpuFfn> CreateQwenNpuFfn(
    const models::QwenModelWeights& weights, std::string* error);

}  // namespace gufo::hip
#endif
#endif  // GUFO_MODELS_QWEN_HIP_NPU_FFN_HPP_
