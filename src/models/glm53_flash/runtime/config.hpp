#ifndef GUFO_MODELS_GLM53_FLASH_RUNTIME_CONFIG_HPP_
#define GUFO_MODELS_GLM53_FLASH_RUNTIME_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::glm53_flash {

/// Hyperparameters of the GLM-5.3-Flash (`glm5-next`) trunk.
///
/// The artifact mixes two attention kinds: KDA layers (`attn_q/k/v` plus the
/// `ssm_*` recurrence) and MLA layers with a DSA indexer. `head_count_kv` is a
/// per-layer flag array in the GGUF: a non-zero entry marks an MLA layer.
struct Config {
  std::uint32_t num_layers{0};        ///< 46, the last one is the MTP block.
  std::uint32_t hidden_size{0};       ///< 4096.
  std::uint32_t num_heads{0};         ///< 64.
  std::uint32_t key_length{0};        ///< 512 (KDA head width).
  std::uint32_t value_length{0};      ///< 512 (KDA head width).
  std::uint32_t key_length_mla{0};    ///< 256.
  std::uint32_t value_length_mla{0};  ///< 256.
  std::uint32_t kv_lora_rank{0};      ///< 512.
  std::uint32_t q_lora_rank{0};       ///< 1536.
  std::uint32_t indexer_heads{0};     ///< 32.
  std::uint32_t indexer_key_dim{0};   ///< 128.
  std::uint32_t indexer_top_k{0};     ///< 2048.
  std::uint32_t indexer_kpool{0};     ///< 4.
  std::uint32_t expert_count{0};      ///< 288.
  std::uint32_t expert_used{0};       ///< 8.
  std::uint32_t expert_shared{0};     ///< 1.
  std::uint32_t expert_ff{0};         ///< 2048.
  std::uint32_t shared_ff{0};         ///< 2048.
  std::uint32_t dense_ff{0};          ///< 12288.
  std::uint32_t leading_dense{0};     ///< 3.
  std::uint32_t hc_count{0};          ///< 4 residual streams.
  std::uint32_t sinkhorn_iters{0};    ///< 20.
  std::uint32_t kda_head_dim{0};      ///< 128.
  std::uint32_t ssm_conv_kernel{0};   ///< 4.
  std::uint32_t vocab_size{0};        ///< 154880.
  std::uint32_t context_length{0};    ///< 1048576.
  std::uint32_t nextn_layers{0};      ///< 1.
  float rms_eps{0.0F};
  float layer_norm_eps{0.0F};
  float hc_eps{0.0F};
  float kda_gate_lower_bound{0.0F};
  float expert_weights_scale{0.0F};
  bool expert_weights_norm{false};
  bool index_share_mtp{false};
  std::vector<std::uint8_t> kv_layer;  ///< `head_count_kv` per layer.
  std::vector<float> swiglu_clamp_exp;
  std::vector<float> swiglu_clamp_shexp;

  [[nodiscard]] bool HasMtp() const noexcept { return nextn_layers != 0; }
  [[nodiscard]] std::uint32_t MtpLayer() const noexcept {
    return HasMtp() ? num_layers - nextn_layers : 0;
  }
  [[nodiscard]] std::uint32_t TrunkLayers() const noexcept {
    return num_layers - nextn_layers;
  }
  [[nodiscard]] bool IsMla(std::uint32_t layer) const noexcept {
    return layer < kv_layer.size() && kv_layer[layer] != 0;
  }
  [[nodiscard]] bool IsKda(std::uint32_t layer) const noexcept {
    return layer < TrunkLayers() && !IsMla(layer);
  }
  [[nodiscard]] bool IsDense(std::uint32_t layer) const noexcept {
    return layer < leading_dense;
  }
  /// Hyper-connection mixer rank: `hc_attn_fn` is [hidden * hc_count, 6 *
  /// hc_count].
  [[nodiscard]] std::uint32_t HcRank() const noexcept { return 6 * hc_count; }
  [[nodiscard]] std::uint32_t HcDim() const noexcept {
    return hc_count * hidden_size;
  }
  [[nodiscard]] std::uint32_t KdaDim() const noexcept {
    return num_heads * kda_head_dim;
  }
  [[nodiscard]] std::uint32_t SsmChannels() const noexcept {
    return 3 * KdaDim();
  }
  [[nodiscard]] std::uint32_t MlaQDim() const noexcept {
    return num_heads * key_length_mla;
  }
  [[nodiscard]] std::uint32_t MlaVDim() const noexcept {
    return num_heads * value_length_mla;
  }
  [[nodiscard]] std::uint32_t IndexerDim() const noexcept {
    return indexer_heads * indexer_key_dim;
  }
  [[nodiscard]] std::uint32_t NumMlaLayers() const noexcept;
  [[nodiscard]] std::uint32_t NumKdaLayers() const noexcept;
};

/// Parses `glm5-next.*` metadata and pins the GLM-5.3-Flash shape. Returns
/// nullopt and a message on any mismatch, so a foreign GLM revision fails at
/// load rather than in the kernels.
[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error_msg);

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_RUNTIME_CONFIG_HPP_
