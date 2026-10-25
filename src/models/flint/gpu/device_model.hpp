#ifndef GUFO_MODELS_FLINT_GPU_DEVICE_MODEL_HPP_
#define GUFO_MODELS_FLINT_GPU_DEVICE_MODEL_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/flint/runtime/config.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::flint::gpu {

/// One weight resident in device memory, still in its GGUF encoding.
/// `rows` is ne[1] and `experts` ne[2], so M and the expert count stay apart.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{1};
  std::uint32_t experts{1};
  /// F16 copy for wide batches (dense and shared projections only).
  void* half_data{nullptr};
};

struct DeviceLayer {
  bool is_moe{false};
  DeviceTensor attn_norm, attn_q, attn_k, attn_v, attn_output;
  /// Q, K and V rows stacked in one Q8_0 matrix; then attn_q/k/v are empty.
  DeviceTensor attn_qkv;
  DeviceTensor attn_q_norm, attn_k_norm, attn_sinks;
  DeviceTensor ffn_norm;
  DeviceTensor ffn_gate, ffn_up, ffn_down;
  DeviceTensor router;
  DeviceTensor gate_exps, up_exps, down_exps;
  DeviceTensor gate_shexp, up_shexp, down_shexp;
  /// Block-float copies of the shared gate and up weights for the NPU.
  std::vector<int> npu_shexp;
  /// The stacked QKV projection, 1024 rows per block-float slice.
  std::vector<int> npu_qkv;
  /// The attention output projection (K = 4096), 1024 rows per slice.
  std::vector<int> npu_out;
};

/// The whole model uploaded to the GPU. Routed down projections stored as
/// Q6_K are re-encoded as Q8_0, the nearest type the routed kernels read.
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const core::GgufReader& reader, const Config& config,
      std::string* error_msg);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceTensor& output_norm() const noexcept { return output_norm_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  std::size_t resident_bytes() const noexcept { return bytes_; }
  qwen38_flash_next::rocm::NpuGemm* npu() const noexcept { return npu_.get(); }
  /// Program id of the K = 4096 instruction stream, or -1.
  int npu_out_program() const noexcept { return npu_out_program_; }
  /// Instruction program and token count of an NPU run covering `n` tokens:
  /// with the runtime-M kernel the smallest multiple of 512 that fits (at
  /// least 1024), otherwise the fixed 4096.
  int NpuProgramFor(std::uint32_t n, std::uint32_t* run_tokens) const noexcept {
    std::uint32_t run = 4096;
    if (npu_m_runtime_) {
      run = std::clamp<std::uint32_t>((n + 511) / 512 * 512, 1024, 4096);
    }
    *run_tokens = run;
    return run == 4096 ? 0 : npu_m_program_[run / 512];
  }

private:
  DeviceModel() = default;
  bool StageNpuWeights();

  Config config_;
  DeviceTensor token_embd_, output_, output_norm_;
  std::vector<DeviceLayer> layers_;
  std::unique_ptr<qwen38_flash_next::rocm::NpuGemm> npu_;
  int npu_out_program_{-1};
  bool npu_m_runtime_{false};
  int npu_m_program_[9]{};
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
};

}  // namespace gufo::models::flint::gpu

#endif  // GUFO_MODELS_FLINT_GPU_DEVICE_MODEL_HPP_
