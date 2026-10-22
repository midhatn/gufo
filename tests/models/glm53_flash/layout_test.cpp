#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/runtime/config.hpp"
#include "src/models/glm53_flash/runtime/layout.hpp"
#include "src/models/glm53_flash/runtime/manifest.hpp"

namespace {

namespace glm = gufo::models::glm53_flash;

int Fail(const std::string& message) {
  std::cerr << "FAIL: " << message << "\n";
  return 1;
}

bool CheckArena(const glm::Layout& layout, glm::Arena arena,
                std::uint64_t total, const std::string& name) {
  std::uint64_t previous_end = 0;
  for (const auto& placement : layout.placements) {
    if (placement.arena != arena) {
      continue;
    }
    if (placement.offset % glm::kArenaAlignment != 0) {
      return Fail(name + ": unaligned offset");
    }
    if (placement.offset < previous_end) {
      return Fail(name + ": overlapping placements");
    }
    previous_end = placement.offset + placement.bytes;
  }
  if (previous_end > total) {
    return Fail(name + ": placements exceed arena size");
  }
  return true;
}

}  // namespace

int main() {
  const char* model_path = std::getenv("GUFO_GLM53_FLASH_MODEL");
  if (model_path == nullptr) {
    return 77;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!reader) {
    return Fail("open: " + error);
  }
  auto config = glm::ParseConfig(*reader, &error);
  if (!config) {
    return Fail("config: " + error);
  }
  const auto& c = *config;
  if (c.TrunkLayers() != 45 || c.NumMlaLayers() != 11 ||
      c.NumKdaLayers() != 34 || c.leading_dense != 3) {
    return Fail("unexpected layer split");
  }

  auto manifest = glm::BuildManifest(c, *reader, &error);
  if (!manifest) {
    return Fail("manifest: " + error);
  }
  if (manifest->entries.size() != reader->GetTensors().size()) {
    return Fail("manifest does not cover every tensor");
  }
  std::uint64_t sum = 0;
  for (const auto& entry : manifest->entries) {
    sum += entry.tensor->size_bytes;
  }
  if (sum != manifest->total_bytes) {
    return Fail("manifest byte total mismatch");
  }

  auto layout = glm::BuildLayout(*manifest, &error);
  if (!layout) {
    return Fail("layout: " + error);
  }
  if (layout->placements.size() != manifest->entries.size()) {
    return Fail("layout entry count mismatch");
  }
  if (!CheckArena(*layout, glm::Arena::kDense, layout->dense_bytes, "dense") ||
      !CheckArena(*layout, glm::Arena::kExperts, layout->expert_bytes,
                  "experts")) {
    return 1;
  }

  std::unordered_map<std::string_view, const glm::ManifestEntry*> by_name;
  for (const auto& entry : manifest->entries) {
    by_name.emplace(entry.tensor->name, &entry);
  }
  const std::size_t tensors = reader->GetTensors().size();
  if (tensors != (c.HasMtp() ? 1412u : 1383u)) {
    return Fail("unexpected tensor count " + std::to_string(tensors));
  }
  const std::string probe = c.HasMtp() ? "blk.45.attn_output.weight"
                                       : "blk.44.attn_output.weight";
  if (!by_name.contains(std::string_view(probe))) {
    return Fail("missing " + probe);
  }
  std::cout << "glm53 layout ok: layers=" << c.num_layers
            << " tensors=" << tensors << " dense_bytes=" << layout->dense_bytes
            << " expert_bytes=" << layout->expert_bytes << "\n";
  return 0;
}
