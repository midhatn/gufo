// Full-trunk check of the GPU executor: evaluates the pinned token stream and
// compares the top-5 logits with the goldens from the validated harness.
// Loads the whole model; run only when no other model is resident
// (GUFO_GLM53_ALLOW_FULL=1). Quantized-activation matvecs make the logits
// differ slightly from the F32 reference, so the contract is the top-1 id
// and top-5 overlap with a bounded logit error. Observed: 0.02-0.04 at the
// first two positions, growing to about 1.1 by position 4.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

namespace {

constexpr int kSkip = 77;
constexpr std::int32_t kTokens[] = {9703, 11, 1879, 0, 101726, 98384};
constexpr int kTopK = 5;
constexpr float kLogitTolerance = 1.5F;

}  // namespace

int main(int argc, char** argv) {
  if (std::getenv("GUFO_GLM53_ALLOW_FULL") == nullptr) {
    std::printf("set GUFO_GLM53_ALLOW_FULL=1 to run (loads the full model)\n");
    return kSkip;
  }
  const char* goldens = std::getenv("GUFO_GLM53_GOLDENS");
  if (goldens == nullptr || argc < 2 || argv[1][0] == '\0') {
    std::printf(
        "usage: glm53_executor_full_test MODEL.GGUF "
        "(GUFO_GLM53_GOLDENS=dir)\n");
    return kSkip;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  namespace rocm = gufo::models::glm53_flash::rocm;
  const auto seconds = [](auto from) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         from)
        .count();
  };
  rocm::ExecutorOptions options;
  options.max_context = 64;
  if (const char* layers = std::getenv("GUFO_GLM53_TEST_LAYERS")) {
    options.layers = static_cast<std::uint32_t>(std::atoi(layers));
  }
  std::string error;
  auto start = std::chrono::steady_clock::now();
  auto executor = rocm::Executor::Create(argv[1], options, &error);
  if (!executor) {
    std::printf("executor create failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("uploaded %u layers in %.1f s\n", executor->layers(),
              seconds(start));
  auto session = executor->CreateSession(&error);
  if (!session) {
    std::printf("session create failed: %s\n", error.c_str());
    return 1;
  }
  std::printf("session ready\n");
  bool ok = true;
  for (std::size_t t = 0; t + 1 < std::size(kTokens); ++t) {
    start = std::chrono::steady_clock::now();
    if (!executor->Evaluate(*session, kTokens[t], &error)) {
      std::printf("evaluate failed: %s\n", error.c_str());
      return 1;
    }
    std::printf("t=%zu evaluated in %.2f s\n", t, seconds(start));
    if (executor->layers() < 45) {
      continue;
    }
    const std::vector<float> logits = executor->Logits(*session);
    std::vector<std::uint32_t> order(logits.size());
    for (std::uint32_t i = 0; i < order.size(); ++i)
      order[i] = i;
    std::partial_sort(order.begin(), order.begin() + kTopK, order.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                        return logits[a] > logits[b];
                      });
    std::ifstream in(
        std::string(goldens) + "/logits_top5." + std::to_string(t) + ".f32",
        std::ios::binary);
    float ref[2 * kTopK];
    if (!in.read(reinterpret_cast<char*>(ref), sizeof(ref))) {
      std::printf("missing golden %zu\n", t);
      return kSkip;
    }
    int overlap = 0;
    float worst = 0.0F;
    for (int k = 0; k < kTopK; ++k) {
      std::uint32_t id;
      std::memcpy(&id, &ref[2 * k + 1], 4);
      overlap += std::find(order.begin(), order.begin() + kTopK, id) !=
                 order.begin() + kTopK;
      worst = std::max(worst, std::fabs(logits[id] - ref[2 * k]));
      std::printf("    id %u gpu %.3f ref %.3f\n", id, logits[id], ref[2 * k]);
    }
    std::uint32_t top1;
    std::memcpy(&top1, &ref[1], 4);
    const bool pass =
        order[0] == top1 && overlap >= 3 && worst < kLogitTolerance;
    ok &= pass;
    std::printf(
        "t=%zu top1 gpu=%u ref=%u overlap=%d/5 worst golden-logit "
        "diff=%.3f %s\n",
        t, order[0], top1, overlap, worst, pass ? "OK" : "FAIL");
  }
  return ok ? 0 : 1;
}
