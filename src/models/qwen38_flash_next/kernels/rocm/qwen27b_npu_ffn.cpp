// NPU offload of the Qwen3.8-27B FFN gate/up projections during prefill (see
// src/models/qwen/hip/npu_ffn.hpp). The qwen executor registers nothing about
// the NPU itself; RegisterQwenNpuFfn() hands it this factory.
#include <hip/hip_fp16.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/qwen/hip/npu_ffn.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/npu_gemm.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kTokens = 4096;
constexpr std::uint32_t kHidden = 5120;
constexpr std::uint32_t kIntermediate = 17408;
constexpr std::uint32_t kSliceColumns = 1024;
constexpr std::uint32_t kDefaultSlices = 6;
constexpr std::uint32_t kIntermediateK = kIntermediate;
constexpr std::uint32_t kDefaultDownSlices = 1;
constexpr std::uint32_t kDefaultQkvSlices = 4;
constexpr std::uint32_t kDefaultGateSlices = 2;
constexpr std::uint32_t kMaxSlices = 16;

bool EncodableType(core::GgmlType type) {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ6_K || type == core::GgmlType::kQ3_K ||
         type == core::GgmlType::kIQ4_NL || type == core::GgmlType::kIQ4_XS ||
         type == core::GgmlType::kIQ3_S;
}

class Qwen27bNpuFfn final : public hip::QwenNpuFfn {
public:
  static std::unique_ptr<hip::QwenNpuFfn> Create(
      const QwenModelWeights& weights, std::string* error) {
    const auto& config = weights.config;
    if (config.hidden_size != kHidden ||
        config.intermediate_size != kIntermediate) {
      return nullptr;
    }
    const char* dir = std::getenv("GUFO_NPU_DIR");
    if (dir == nullptr) {
      return nullptr;
    }
    std::uint32_t slices = kDefaultSlices;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_FFN_SLICES")) {
      slices = static_cast<std::uint32_t>(
          std::clamp(std::atoi(v), 0, static_cast<int>(kMaxSlices)));
    }
    std::uint32_t qkv_slices = kDefaultQkvSlices;
    std::uint32_t gate_slices = kDefaultGateSlices;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_QKV_SLICES")) {
      qkv_slices = static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 10));
    }
    if (const char* v = std::getenv("GUFO_QWEN_NPU_GATE_SLICES")) {
      gate_slices = static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 6));
    }
    std::uint32_t down_slices = kDefaultDownSlices;
    if (const char* v = std::getenv("GUFO_QWEN_NPU_DOWN_SLICES")) {
      down_slices = static_cast<std::uint32_t>(std::clamp(std::atoi(v), 0, 5));
    }
    if (slices == 0 && qkv_slices + gate_slices + down_slices == 0) {
      return nullptr;
    }
    NpuGemm::Config cfg;
    cfg.xclbin =
        std::string(dir) + "/runtime_k/final_4096xKx1024_128x64x128.xclbin";
    cfg.insts =
        std::string(dir) + "/runtime_k/insts_4096x5120x1024_128x64x128.txt";
    cfg.tokens = kTokens;
    cfg.k = kHidden;
    cfg.n = kSliceColumns;
    auto npu = NpuGemm::Create(cfg, error);
    if (npu == nullptr) {
      return nullptr;
    }
    auto ffn = std::unique_ptr<Qwen27bNpuFfn>(new Qwen27bNpuFfn());
    ffn->slices_ = slices;
    ffn->qkv_slices_ = qkv_slices;
    ffn->gate_slices_ = gate_slices;
    ffn->down_slices_ = down_slices;
    if (down_slices != 0) {
      ffn->down_program_ = npu->AddProgram(
          std::string(dir) + "/runtime_k/insts_4096x17408x1024_128x64x128.txt",
          error);
      if (ffn->down_program_ < 0) {
        return nullptr;
      }
      ffn->a_down_ = npu->NewBuffer(npu->ABytes(kIntermediateK), error);
      if (ffn->a_down_ < 0) {
        return nullptr;
      }
    }
    ffn->a_ = npu->NewBuffer(npu->ABytes(), error);
    if (ffn->a_ < 0) {
      return nullptr;
    }
    for (std::uint32_t i = 0;
         i < std::max({2 * slices, qkv_slices + gate_slices, down_slices});
         ++i) {
      const int id = npu->NewBuffer(npu->CBytes(), error);
      if (id < 0) {
        return nullptr;
      }
      ffn->c_.push_back(id);
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
      return nullptr;
    }
    std::size_t staged = 0;
    std::uint32_t layers_on_npu = 0;
    ffn->b_.resize(weights.layers.size());
    ffn->columns_.assign(weights.layers.size(), 0);
    ffn->lin_b_.resize(weights.layers.size());
    ffn->down_b_.resize(weights.layers.size());
    ffn->down_columns_.assign(weights.layers.size(), 0);
    ffn->qkv_columns_.assign(weights.layers.size(), 0);
    ffn->gate_columns_.assign(weights.layers.size(), 0);
    const auto encode = [&](const QwenTensorRef& w, std::uint32_t count,
                            std::vector<int>& ids, std::uint32_t k = kHidden) {
      for (std::uint32_t i = 0; i < count; ++i) {
        const int id = npu->NewBuffer(npu->BBytes(k), error);
        if (id < 0) {
          return false;
        }
        NpuEncodeWeightsQuant(w.data, static_cast<std::uint32_t>(w.type), k,
                              i * kSliceColumns, kSliceColumns, k, 0, k,
                              npu->Device(id), stream);
        ids.push_back(id);
        staged += npu->BBytes(k);
      }
      return true;
    };
    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
      const auto& layer = weights.layers[l];
      std::vector<int> ids;
      bool ok = true;
      if (slices != 0 && EncodableType(layer.ffn_gate.type) &&
          EncodableType(layer.ffn_up.type)) {
        ok = encode(layer.ffn_gate, slices, ids) &&
             encode(layer.ffn_up, slices, ids);
        if (ok) {
          ffn->columns_[l] = slices * kSliceColumns;
          ++layers_on_npu;
        }
      }
      std::vector<int> lin;
      if (ok && !layer.is_full_attention && qkv_slices + gate_slices != 0 &&
          EncodableType(layer.attn_qkv.type) &&
          EncodableType(layer.attn_gate.type) &&
          layer.attn_qkv.num_elements == std::size_t{10240} * kHidden &&
          layer.attn_gate.num_elements == std::size_t{6144} * kHidden) {
        ok = encode(layer.attn_qkv, qkv_slices, lin) &&
             encode(layer.attn_gate, gate_slices, lin);
        if (ok) {
          ffn->qkv_columns_[l] = qkv_slices * kSliceColumns;
          ffn->gate_columns_[l] = gate_slices * kSliceColumns;
        }
      }
      std::vector<int> down;
      if (ok && down_slices != 0 && EncodableType(layer.ffn_down.type) &&
          layer.ffn_down.num_elements ==
              std::size_t{kHidden} * kIntermediateK) {
        ok = encode(layer.ffn_down, down_slices, down, kIntermediateK);
        if (ok) {
          ffn->down_columns_[l] = down_slices * kSliceColumns;
        }
      }
      if (!ok) {
        hipStreamDestroy(stream);
        return nullptr;
      }
      hipStreamSynchronize(stream);
      for (const int id : down) {
        npu->FinalizeWeights(id);
      }
      ffn->down_b_[l] = std::move(down);
      for (const int id : ids) {
        npu->FinalizeWeights(id);
      }
      for (const int id : lin) {
        npu->FinalizeWeights(id);
      }
      ffn->b_[l] = std::move(ids);
      ffn->lin_b_[l] = std::move(lin);
    }
    hipStreamDestroy(stream);
    std::fprintf(stderr,
                 "NPU: FFN gate/up first %u columns of %u layers, linear qkv "
                 "%u gate %u, down %u columns, %.2f GiB staged\n",
                 slices * kSliceColumns, layers_on_npu,
                 qkv_slices * kSliceColumns, gate_slices * kSliceColumns,
                 down_slices * kSliceColumns,
                 static_cast<double>(staged) / (1024.0 * 1024.0 * 1024.0));
    ffn->npu_ = std::move(npu);
    return ffn;
  }

  std::uint32_t Tokens() const noexcept override { return kTokens; }
  std::uint32_t Columns(std::uint32_t layer) const noexcept override {
    return layer < columns_.size() ? columns_[layer] : 0;
  }

  bool Begin(std::uint32_t layer, const void* normed,
             hipStream_t stream) override {
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(normed), kHidden, kTokens,
                         kHidden, 0, kHidden, npu_->Device(a_), stream);
    const auto& b = b_[layer];
    if (!npu_->Submit(stream, a_, b.data(), c_.data(),
                      static_cast<int>(b.size()))) {
      Release(this);
      return false;
    }
    return true;
  }

  void Finish(std::uint32_t layer, void* out, std::size_t ld,
              hipStream_t stream) override {
    (void)layer;
    npu_->Wait(stream);
    for (std::uint32_t i = 0; i < slices_; ++i) {
      NpuSwiglu(npu_->Staged(stream, c_[i]),
                npu_->Staged(stream, c_[slices_ + i]), nullptr, nullptr,
                static_cast<__half*>(out), kTokens, kSliceColumns,
                kSliceColumns, stream, ld, i * kSliceColumns);
    }
    // The shared buffers are free once these kernels have read them.
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

  std::uint32_t DownColumns(std::uint32_t layer) const noexcept override {
    return layer < down_columns_.size() ? down_columns_[layer] : 0;
  }

  bool BeginDown(std::uint32_t layer, const void* act,
                 hipStream_t stream) override {
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(act), kIntermediateK,
                         kTokens, kIntermediateK, 0, kIntermediateK,
                         npu_->Device(a_down_), stream);
    const auto& b = down_b_[layer];
    if (!npu_->Submit(stream, a_down_, b.data(), c_.data(),
                      static_cast<int>(b.size()), down_program_)) {
      Release(this);
      return false;
    }
    return true;
  }

  void FinishDown(std::uint32_t layer, float* residual, std::size_t ld,
                  hipStream_t stream) override {
    npu_->Wait(stream);
    const std::uint32_t count = down_columns_[layer] / kSliceColumns;
    for (std::uint32_t i = 0; i < count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, c_[i]), residual, kTokens,
                      kSliceColumns, ld, i * kSliceColumns, true, stream);
    }
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

  std::uint32_t QkvColumns(std::uint32_t layer) const noexcept override {
    return layer < qkv_columns_.size() ? qkv_columns_[layer] : 0;
  }
  std::uint32_t GateColumns(std::uint32_t layer) const noexcept override {
    return layer < gate_columns_.size() ? gate_columns_[layer] : 0;
  }

  bool BeginLinear(std::uint32_t layer, const void* normed,
                   hipStream_t stream) override {
    if (!npu_->Healthy() || !Acquire(stream)) {
      return false;
    }
    NpuEncodeActivations(static_cast<const __half*>(normed), kHidden, kTokens,
                         kHidden, 0, kHidden, npu_->Device(a_), stream);
    const auto& b = lin_b_[layer];
    if (!npu_->Submit(stream, a_, b.data(), c_.data(),
                      static_cast<int>(b.size()))) {
      Release(this);
      return false;
    }
    return true;
  }

  void FinishLinear(std::uint32_t layer, float* qkv, std::size_t qkv_ld,
                    float* gate, std::size_t gate_ld,
                    hipStream_t stream) override {
    npu_->Wait(stream);
    const std::uint32_t qkv_count = qkv_columns_[layer] / kSliceColumns;
    const std::uint32_t gate_count = gate_columns_[layer] / kSliceColumns;
    for (std::uint32_t i = 0; i < qkv_count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, c_[i]), qkv, kTokens, kSliceColumns,
                      qkv_ld, i * kSliceColumns, false, stream);
    }
    for (std::uint32_t i = 0; i < gate_count; ++i) {
      NpuDecodeOutput(npu_->Staged(stream, c_[qkv_count + i]), gate, kTokens,
                      kSliceColumns, gate_ld, i * kSliceColumns, false, stream);
    }
    hipLaunchHostFunc(
        stream, [](void* self) { Release(static_cast<Qwen27bNpuFfn*>(self)); },
        this);
  }

private:
  Qwen27bNpuFfn() = default;

  /// One stream owns the shared NPU buffers at a time. Its own layers follow
  /// each other in stream order, so they need no host-side wait; another
  /// stream is refused until the owner's last queued layer has finished.
  bool Acquire(hipStream_t stream) {
    hipStream_t expected = nullptr;
    if (!owner_.compare_exchange_strong(expected, stream) &&
        expected != stream) {
      return false;
    }
    pending_.fetch_add(1);
    return true;
  }
  static void Release(Qwen27bNpuFfn* self) {
    if (self->pending_.fetch_sub(1) == 1) {
      self->owner_.store(nullptr);
    }
  }

  std::unique_ptr<NpuGemm> npu_;
  std::uint32_t slices_{0};
  std::uint32_t down_slices_{0};
  int down_program_{0};
  int a_down_{-1};
  std::vector<std::vector<int>> down_b_;
  std::vector<std::uint32_t> down_columns_;
  std::uint32_t qkv_slices_{0};
  std::uint32_t gate_slices_{0};
  std::vector<std::vector<int>> lin_b_;
  std::vector<std::uint32_t> qkv_columns_;
  std::vector<std::uint32_t> gate_columns_;
  int a_{-1};
  std::vector<int> c_;
  std::vector<std::vector<int>> b_;
  std::vector<std::uint32_t> columns_;
  std::atomic<hipStream_t> owner_{nullptr};
  std::atomic<int> pending_{0};
};

}  // namespace

void RegisterQwenNpuFfn() {
  hip::SetQwenNpuFfnFactory(&Qwen27bNpuFfn::Create);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
