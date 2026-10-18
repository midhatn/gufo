#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/npu_ffn.hpp"

#include <utility>

namespace gufo::hip {
namespace {
QwenNpuFfnFactory& Factory() {
  static QwenNpuFfnFactory factory;
  return factory;
}
}  // namespace

void SetQwenNpuFfnFactory(QwenNpuFfnFactory factory) {
  Factory() = std::move(factory);
}

std::unique_ptr<QwenNpuFfn> CreateQwenNpuFfn(
    const models::QwenModelWeights& weights, std::string* error) {
  const auto& factory = Factory();
  return factory ? factory(weights, error) : nullptr;
}

}  // namespace gufo::hip
#endif
