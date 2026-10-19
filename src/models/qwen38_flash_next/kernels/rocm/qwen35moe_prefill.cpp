// Grouped-expert prefill for Qwen3.6-35B-A3B (qwen35moe), registered with the
// qwen executor (src/models/qwen/hip/moe_prefill.hpp). It drives the routed
// expert kernels of the Flash-Next library: top-k routing, expert-sorted row
// compaction, paired gate/up and down GEMMs on the F16 matrix cores and the
// combine with the shared expert.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "src/models/qwen/hip/moe_prefill.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

constexpr std::uint32_t kPairTileRows = 128;
constexpr std::uint32_t kPlainTileRows = 48;
constexpr std::uint32_t kQ8TileRows = 64;
constexpr std::uint32_t kPadRows = 16;

struct LayerExperts {
  bool supported{false};
  const void* gate{nullptr};
  const void* up{nullptr};
  const void* down{nullptr};
  WeightType gate_type{WeightType::kQ4_K};
  WeightType down_type{WeightType::kQ5_K};
  std::uint32_t down_tile_rows{kPlainTileRows};
  void* converted_down{nullptr};
};

bool HipOk(hipError_t status, const char* what, std::string* error) {
  if (status == hipSuccess) {
    return true;
  }
  if (error != nullptr) {
    *error = std::string(what) + ": " + hipGetErrorString(status);
  }
  return false;
}

class Qwen35MoePrefill final : public hip::QwenMoePrefill {
public:
  static std::unique_ptr<hip::QwenMoePrefill> Create(
      const QwenModelWeights& weights, std::uint32_t max_tokens,
      std::string* error) {
    const auto& c = weights.config;
    if (!c.IsMoe() || c.hidden_size % 256 != 0 || c.expert_ff % 256 != 0) {
      return nullptr;
    }
    std::unique_ptr<Qwen35MoePrefill> self(new Qwen35MoePrefill());
    self->config_ = c;
    self->max_tokens_ = max_tokens;
    if (!self->Allocate(error)) {
      return nullptr;
    }
    hipStream_t stream = nullptr;
    if (!HipOk(hipStreamCreate(&stream), "stream", error)) {
      return nullptr;
    }
    std::size_t converted_bytes = 0;
    self->layers_.resize(weights.layers.size());
    for (std::size_t l = 0; l < weights.layers.size(); ++l) {
      const auto& layer = weights.layers[l];
      LayerExperts& out = self->layers_[l];
      const bool gate_ok = layer.ffn_gate_exps.type == core::GgmlType::kQ4_K ||
                           layer.ffn_gate_exps.type == core::GgmlType::kQ5_K;
      if (!gate_ok || layer.ffn_up_exps.type != layer.ffn_gate_exps.type) {
        continue;
      }
      out.gate = layer.ffn_gate_exps.data;
      out.up = layer.ffn_up_exps.data;
      out.gate_type = layer.ffn_gate_exps.type == core::GgmlType::kQ5_K
                          ? WeightType::kQ5_K
                          : WeightType::kQ4_K;
      switch (layer.ffn_down_exps.type) {
        case core::GgmlType::kQ5_K:
          out.down = layer.ffn_down_exps.data;
          out.down_type = WeightType::kQ5_K;
          out.down_tile_rows = kPlainTileRows;
          break;
        case core::GgmlType::kQ8_0:
          out.down = layer.ffn_down_exps.data;
          out.down_type = WeightType::kQ8_0;
          out.down_tile_rows = kQ8TileRows;
          break;
        case core::GgmlType::kQ6_K: {
          // The grouped down kernel has no Q6_K decoder: keep an int8 copy.
          const std::size_t rows = std::size_t{c.num_experts} * c.hidden_size;
          const std::size_t bytes = rows * (c.expert_ff / 32) * 34;
          if (!HipOk(hipMalloc(&out.converted_down, bytes), "q8 copy", error)) {
            hipStreamDestroy(stream);
            return nullptr;
          }
          RequantToQ8_0(layer.ffn_down_exps.data,
                        static_cast<std::uint32_t>(layer.ffn_down_exps.type),
                        c.expert_ff, rows, out.converted_down, stream);
          converted_bytes += bytes;
          out.down = out.converted_down;
          out.down_type = WeightType::kQ8_0;
          out.down_tile_rows = kQ8TileRows;
          break;
        }
        default:
          continue;
      }
      out.supported = true;
    }
    const bool synced = HipOk(hipStreamSynchronize(stream), "sync", error);
    hipStreamDestroy(stream);
    if (!synced) {
      return nullptr;
    }
    if (converted_bytes != 0) {
      std::fprintf(
          stderr, "MoE prefill: %.2f GiB of Q6_K experts re-encoded as Q8_0\n",
          static_cast<double>(converted_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    return self;
  }

  ~Qwen35MoePrefill() override {
    for (auto& layer : layers_) {
      if (layer.converted_down != nullptr) {
        (void)hipFree(layer.converted_down);
      }
    }
    for (void* pointer :
         {static_cast<void*>(ids_), static_cast<void*>(weights_),
          static_cast<void*>(counts_), static_cast<void*>(bounds_),
          static_cast<void*>(cursors_), static_cast<void*>(rows_token_),
          static_cast<void*>(rows_slot_), static_cast<void*>(tiles_pair_),
          static_cast<void*>(tiles_down_), act_, down_}) {
      if (pointer != nullptr) {
        (void)hipFree(pointer);
      }
    }
    if (host_counts_ != nullptr) {
      (void)hipHostFree(host_counts_);
    }
    if (host_tiles_ != nullptr) {
      (void)hipHostFree(host_tiles_);
    }
  }

  std::uint32_t MaxTokens() const noexcept override { return max_tokens_; }
  bool Supports(std::uint32_t layer) const noexcept override {
    return layer < layers_.size() && layers_[layer].supported;
  }

  bool Run(std::uint32_t layer, std::uint32_t n_tokens,
           const float* router_logits, const __half* x_half,
           const float* shared_out, const float* shared_logit, float* out,
           hipStream_t stream) override {
    if (!Supports(layer) || n_tokens == 0 || n_tokens > max_tokens_) {
      return false;
    }
    const LayerExperts& experts = layers_[layer];
    const std::uint32_t used = config_.num_experts_used;
    const std::uint32_t num_experts = config_.num_experts;
    const std::uint32_t hidden = config_.hidden_size;
    const std::uint32_t ff = config_.expert_ff;
    RouterTopK(router_logits, num_experts, ids_, weights_, n_tokens,
               num_experts, used, stream);
    ExpertCounts(ids_, counts_, n_tokens, num_experts, used, stream);
    if (hipMemcpyAsync(host_counts_, counts_,
                       num_experts * sizeof(std::uint32_t),
                       hipMemcpyDeviceToHost, stream) != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
      return false;
    }
    // Tile maps: (expert | tile << 16) per row tile of an expert's 16-padded
    // bucket, one map per tile width.
    std::uint32_t pair_tiles = 0;
    std::uint32_t down_tiles = 0;
    std::int32_t* host_pair = host_tiles_;
    std::int32_t* host_down = host_tiles_ + tile_capacity_;
    for (std::uint32_t e = 0; e < num_experts; ++e) {
      const std::uint32_t padded =
          (host_counts_[e] + kPadRows - 1) / kPadRows * kPadRows;
      for (std::uint32_t j = 0;
           j < (padded + kPairTileRows - 1) / kPairTileRows; ++j) {
        host_pair[pair_tiles++] = static_cast<std::int32_t>(e | (j << 16));
      }
      for (std::uint32_t j = 0;
           j < (padded + experts.down_tile_rows - 1) / experts.down_tile_rows;
           ++j) {
        host_down[down_tiles++] = static_cast<std::int32_t>(e | (j << 16));
      }
    }
    if (pair_tiles == 0) {
      return false;
    }
    if (hipMemcpyAsync(tiles_pair_, host_pair,
                       pair_tiles * sizeof(std::int32_t), hipMemcpyHostToDevice,
                       stream) != hipSuccess ||
        hipMemcpyAsync(tiles_down_, host_down,
                       down_tiles * sizeof(std::int32_t), hipMemcpyHostToDevice,
                       stream) != hipSuccess) {
      return false;
    }
    RoutedCompact(ids_, counts_, bounds_, cursors_, rows_token_, rows_slot_,
                  n_tokens, used, num_experts, stream);
    auto* act = static_cast<__half*>(act_);
    if (!RoutedGatedF16Gemm(experts.gate, experts.up, experts.gate_type, x_half,
                            tiles_pair_, pair_tiles, kPairTileRows, bounds_,
                            rows_token_, rows_slot_, act, ff, hidden, stream)) {
      return false;
    }
    auto* down = static_cast<__half*>(down_);
    if (!RoutedF16Gemm(experts.down, experts.down_type, act, tiles_down_,
                       down_tiles, experts.down_tile_rows, bounds_, rows_slot_,
                       rows_slot_, nullptr, nullptr, down, hidden, ff,
                       stream)) {
      return false;
    }
    MoeEpilogueVec4F16(down, weights_, shared_out, shared_logit, 1, out,
                       n_tokens, used, hidden, stream);
    return true;
  }

private:
  Qwen35MoePrefill() = default;

  bool Allocate(std::string* error) {
    const std::size_t slots =
        std::size_t{max_tokens_} * config_.num_experts_used;
    const std::size_t rows = RoutedCompactRows(slots, config_.num_experts);
    tile_capacity_ = static_cast<std::uint32_t>(
        (slots + std::size_t{config_.num_experts} * (kPadRows - 1)) / kPadRows +
        config_.num_experts + 1);
    return HipOk(hipMalloc(&ids_, slots * sizeof(std::int32_t)), "ids",
                 error) &&
           HipOk(hipMalloc(&weights_, slots * sizeof(float)), "weights",
                 error) &&
           HipOk(
               hipMalloc(&counts_, config_.num_experts * sizeof(std::uint32_t)),
               "counts", error) &&
           HipOk(hipMalloc(&bounds_,
                           (config_.num_experts + 1) * sizeof(std::int32_t)),
                 "bounds", error) &&
           HipOk(
               hipMalloc(&cursors_, config_.num_experts * sizeof(std::int32_t)),
               "cursors", error) &&
           HipOk(hipMalloc(&rows_token_, rows * sizeof(std::int32_t)), "rows",
                 error) &&
           HipOk(hipMalloc(&rows_slot_, rows * sizeof(std::int32_t)), "rows",
                 error) &&
           HipOk(hipMalloc(&tiles_pair_, tile_capacity_ * sizeof(std::int32_t)),
                 "tiles", error) &&
           HipOk(hipMalloc(&tiles_down_, tile_capacity_ * sizeof(std::int32_t)),
                 "tiles", error) &&
           HipOk(hipMalloc(&act_, slots * config_.expert_ff * sizeof(__half)),
                 "act", error) &&
           HipOk(
               hipMalloc(&down_, slots * config_.hidden_size * sizeof(__half)),
               "down", error) &&
           HipOk(hipHostMalloc(reinterpret_cast<void**>(&host_counts_),
                               config_.num_experts * sizeof(std::uint32_t)),
                 "host counts", error) &&
           HipOk(hipHostMalloc(reinterpret_cast<void**>(&host_tiles_),
                               2 * tile_capacity_ * sizeof(std::int32_t)),
                 "host tiles", error);
  }

  core::ModelConfig config_;
  std::uint32_t max_tokens_{0};
  std::uint32_t tile_capacity_{0};
  std::vector<LayerExperts> layers_;
  std::int32_t* ids_{nullptr};
  float* weights_{nullptr};
  std::uint32_t* counts_{nullptr};
  std::int32_t* bounds_{nullptr};
  std::int32_t* cursors_{nullptr};
  std::int32_t* rows_token_{nullptr};
  std::int32_t* rows_slot_{nullptr};
  std::int32_t* tiles_pair_{nullptr};
  std::int32_t* tiles_down_{nullptr};
  void* act_{nullptr};
  void* down_{nullptr};
  std::uint32_t* host_counts_{nullptr};
  std::int32_t* host_tiles_{nullptr};
};

}  // namespace

void RegisterQwenMoePrefill() {
  hip::SetQwenMoePrefillFactory(&Qwen35MoePrefill::Create);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
