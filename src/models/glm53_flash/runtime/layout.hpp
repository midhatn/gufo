#ifndef GUFO_MODELS_GLM53_FLASH_RUNTIME_LAYOUT_HPP_
#define GUFO_MODELS_GLM53_FLASH_RUNTIME_LAYOUT_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/runtime/manifest.hpp"

namespace gufo::models::glm53_flash {

/// Byte alignment of every tensor inside a device arena. All ggml block
/// formats divide it, so a single arena can mix quantized and float tensors.
inline constexpr std::uint64_t kArenaAlignment = 256;

/// Where a manifest entry lives. Routed experts are split into their own arena
/// because they are the only tensors that may be tiered out of device memory.
enum class Arena : std::uint8_t { kDense, kExperts };

struct WeightPlacement {
  Slot slot{Slot::kTokenEmbd};
  std::uint32_t layer{0};
  Arena arena{Arena::kDense};
  std::uint64_t offset{0};
  std::uint64_t bytes{0};
  core::GgmlType type{core::GgmlType::kF32};
};

struct Layout {
  std::vector<WeightPlacement> placements;
  std::uint64_t dense_bytes{0};
  std::uint64_t expert_bytes{0};
  [[nodiscard]] std::uint64_t TotalBytes() const noexcept {
    return dense_bytes + expert_bytes;
  }
};

/// Assigns every tensor a 256-byte aligned slot in the dense or expert arena and
/// verifies that the on-disk size matches the declared type and shape.
[[nodiscard]] std::optional<Layout> BuildLayout(const Manifest& manifest,
                                                std::string* error_msg);

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_RUNTIME_LAYOUT_HPP_
