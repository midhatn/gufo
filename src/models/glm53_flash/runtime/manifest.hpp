#ifndef GUFO_MODELS_GLM53_FLASH_RUNTIME_MANIFEST_HPP_
#define GUFO_MODELS_GLM53_FLASH_RUNTIME_MANIFEST_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "src/models/glm53_flash/runtime/config.hpp"

namespace gufo::core {
class GgufReader;
struct GgufTensorInfo;
}  // namespace gufo::core

namespace gufo::models::glm53_flash {

/// Logical slot of a tensor in the `glm5-next` graph. The manifest pins every
/// name/dimension so a repack that renames or reshapes a tensor fails at load.
enum class Slot : std::uint8_t {
  kTokenEmbd,
  kOutput,
  kOutputNorm,
  kAttnNorm,
  kFfnNorm,
  kHcAttnBase,
  kHcAttnFn,
  kHcAttnScale,
  kHcFfnBase,
  kHcFfnFn,
  kHcFfnScale,
  kKdaQ,
  kKdaK,
  kKdaV,
  kKdaOutput,
  kKdaSsmA,
  kKdaBeta,
  kKdaDt,
  kKdaConvQ,
  kKdaConvK,
  kKdaConvV,
  kKdaFA,
  kKdaFB,
  kKdaGA,
  kKdaGB,
  kKdaNorm,
  kDenseGate,
  kDenseUp,
  kDenseDown,
  kMlaQa,
  kMlaQaNorm,
  kMlaQb,
  kMlaKvA,
  kMlaKvANorm,
  kMlaKb,
  kMlaVb,
  kMlaOutput,
  kIndexerK,
  kIndexerQb,
  kIndexerKNorm,
  kIndexerKNormBias,
  kIndexerProj,
  kIndexerApe,
  kIndexerGate,
  kRouter,
  kRouterBias,
  kExpertGate,
  kExpertUp,
  kExpertDown,
  kSharedGate,
  kSharedUp,
  kSharedDown,
  kNextnEhProj,
  kNextnEnorm,
  kNextnHnorm,
  kNextnSharedHeadNorm,
};

/// True when the slot stores 32-bit floats on disk; everything else must be a
/// block-quantized format the matmul backend understands.
[[nodiscard]] bool IsFloatSlot(Slot slot) noexcept;

struct ManifestEntry {
  Slot slot{Slot::kTokenEmbd};
  std::uint32_t layer{0};
  const core::GgufTensorInfo* tensor{nullptr};
};

struct Manifest {
  std::vector<ManifestEntry> entries;
  std::uint64_t total_bytes{0};
  std::uint64_t float_bytes{0};
  std::uint64_t quantized_bytes{0};
  std::uint64_t expert_bytes{0};
  std::uint64_t mtp_bytes{0};
};

/// Builds and verifies the manifest for the pinned GLM-5.3-Flash layout.
[[nodiscard]] std::optional<Manifest> BuildManifest(
    const Config& config, const core::GgufReader& reader, std::string* error_msg);

}  // namespace gufo::models::glm53_flash

#endif  // GUFO_MODELS_GLM53_FLASH_RUNTIME_MANIFEST_HPP_
