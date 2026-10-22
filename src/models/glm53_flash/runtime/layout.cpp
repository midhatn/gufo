#include "src/models/glm53_flash/runtime/layout.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::glm53_flash {
namespace {

Arena ArenaOf(Slot slot) noexcept {
  switch (slot) {
    case Slot::kExpertGate:
    case Slot::kExpertUp:
    case Slot::kExpertDown:
      return Arena::kExperts;
    default:
      return Arena::kDense;
  }
}

std::uint64_t Align(std::uint64_t value) noexcept {
  return (value + kArenaAlignment - 1) / kArenaAlignment * kArenaAlignment;
}

}  // namespace

std::optional<Layout> BuildLayout(const Manifest& manifest,
                                  std::string* error_msg) {
  Layout layout;
  layout.placements.reserve(manifest.entries.size());
  const auto fail = [error_msg](std::string message) -> std::optional<Layout> {
    if (error_msg != nullptr) {
      *error_msg = std::move(message);
    }
    return std::nullopt;
  };

  for (const auto& entry : manifest.entries) {
    const auto* tensor = entry.tensor;
    const auto elements = tensor->ElementCount();
    const auto encoded =
        quant::EncodedSizeBytes(tensor->type, static_cast<std::size_t>(elements));
    if (encoded == 0) {
      return fail("cannot size " + std::string(tensor->name) + " as " +
                  std::string(core::ToString(tensor->type)));
    }
    if (encoded != tensor->size_bytes) {
      return fail(std::string(tensor->name) + " occupies " +
                  std::to_string(tensor->size_bytes) + " bytes on disk but " +
                  std::to_string(encoded) + " are implied by its shape");
    }
    const Arena arena = ArenaOf(entry.slot);
    auto& cursor = arena == Arena::kExperts ? layout.expert_bytes
                                            : layout.dense_bytes;
    cursor = Align(cursor);
    if (cursor > std::numeric_limits<std::uint64_t>::max() - encoded) {
      return fail("arena size overflow at " + std::string(tensor->name));
    }
    layout.placements.push_back(WeightPlacement{
        entry.slot, entry.layer, arena, cursor, encoded, tensor->type});
    cursor += encoded;
  }
  layout.dense_bytes = Align(layout.dense_bytes);
  layout.expert_bytes = Align(layout.expert_bytes);
  return layout;
}

}  // namespace gufo::models::glm53_flash
