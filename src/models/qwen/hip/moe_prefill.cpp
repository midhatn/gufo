#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_prefill.hpp"

#include <utility>

namespace gufo::hip {
namespace {
QwenMoePrefillFactory& Factory() {
  static QwenMoePrefillFactory factory;
  return factory;
}
}  // namespace

void SetQwenMoePrefillFactory(QwenMoePrefillFactory factory) {
  Factory() = std::move(factory);
}

std::unique_ptr<QwenMoePrefill> CreateQwenMoePrefill(
    const models::QwenModelWeights& weights, std::uint32_t max_tokens,
    std::string* error) {
  const auto& factory = Factory();
  return factory ? factory(weights, max_tokens, error) : nullptr;
}

}  // namespace gufo::hip
#endif
