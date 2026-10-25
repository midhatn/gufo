#ifndef GUFO_MODELS_FLINT_RUNTIME_CONFIG_HPP_
#define GUFO_MODELS_FLINT_RUNTIME_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::flint {

struct Config {
  std::uint32_t num_layers{0};
  std::uint32_t context_length{0};
  std::uint32_t hidden_size{0};
  std::uint32_t dense_ff{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv{0};
  std::uint32_t head_dim{0};
  std::uint32_t vocab_size{0};
  std::uint32_t expert_count{0};
  std::uint32_t expert_used{0};
  std::uint32_t expert_ff{0};
  std::uint32_t shared_ff{0};
  std::uint32_t leading_dense{0};
  std::uint32_t expert_shared{0};
  std::uint32_t sliding_window{0};
  float rms_eps{0.0F};
  float rope_base{0.0F};
  float expert_weights_scale{1.0F};
  bool expert_weights_norm{true};
  std::uint32_t expert_gating_func{0};
  std::vector<bool> is_swa;
  std::vector<bool> rope_pattern;

  [[nodiscard]] bool HasRope(std::uint32_t layer) const {
    return layer < rope_pattern.size() && rope_pattern[layer];
  }
  [[nodiscard]] bool IsSwa(std::uint32_t layer) const {
    return layer < is_swa.size() && is_swa[layer];
  }
  [[nodiscard]] std::uint32_t NumMoeLayers() const {
    return num_layers > leading_dense ? num_layers - leading_dense : 0;
  }
};

[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error_msg);

}  // namespace gufo::models::flint

#endif  // GUFO_MODELS_FLINT_RUNTIME_CONFIG_HPP_
