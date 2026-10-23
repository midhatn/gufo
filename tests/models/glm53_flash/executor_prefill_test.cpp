// Batched prefill against token-by-token decode on the same executor:
//   glm53_executor_prefill_test MODEL TOKENS.txt
// Both paths must land on near-identical final logits (they differ only in
// the GEMM versus matvec activation rounding), and prefill must be faster.
// GUFO_GLM53_TEST_LAYERS limits the trunk, GUFO_GLM53_TOPPOOLS caps the DSA
// window so short streams exercise selection, GUFO_GLM53_CHUNK sets the chunk.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

namespace {

constexpr int kSkip = 77;
// Rounding differences between the GEMM and matvec paths are amplified by
// near-tied expert routing through 45 layers (8 layers agree at 0.9997).
constexpr double kMinCosine = 0.98;

double Seconds(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - from)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argv[1][0] == '\0') {
    std::printf("usage: %s MODEL TOKENS.txt\n", argv[0]);
    return kSkip;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::vector<std::int32_t> tokens;
  {
    std::ifstream in(argv[2]);
    long v;
    while (in >> v)
      tokens.push_back(static_cast<std::int32_t>(v));
  }
  namespace rocm = gufo::models::glm53_flash::rocm;
  rocm::ExecutorOptions options;
  options.max_context = static_cast<std::uint32_t>(tokens.size()) + 8;
  if (const char* v = std::getenv("GUFO_GLM53_TEST_LAYERS"))
    options.layers = std::atoi(v);
  if (const char* v = std::getenv("GUFO_GLM53_TOPPOOLS"))
    options.top_pools = std::atoi(v);
  if (const char* v = std::getenv("GUFO_GLM53_CHUNK"))
    options.prefill_chunk = std::atoi(v);
  std::string error;
  auto executor = rocm::Executor::Create(argv[1], options, &error);
  if (!executor) {
    std::printf("create failed: %s\n", error.c_str());
    return 1;
  }

  auto seq = executor->CreateSession(&error);
  auto start = std::chrono::steady_clock::now();
  for (const auto token : tokens) {
    if (!executor->Evaluate(*seq, token, &error)) {
      std::printf("evaluate failed: %s\n", error.c_str());
      return 1;
    }
  }
  const double t_seq = Seconds(start);
  const auto a = executor->Logits(*seq);
  const auto ra = executor->Residual(*seq);

  auto bat = executor->CreateSession(&error);
  start = std::chrono::steady_clock::now();
  if (!executor->Prefill(*bat, tokens, &error)) {
    std::printf("prefill failed: %s\n", error.c_str());
    return 1;
  }
  const double t_bat = Seconds(start);
  const auto b = executor->Logits(*bat);
  const auto rb = executor->Residual(*bat);

  const auto cosine = [](const std::vector<float>& x,
                         const std::vector<float>& y) {
    double d = 0, xx = 0, yy = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      d += static_cast<double>(x[i]) * y[i];
      xx += static_cast<double>(x[i]) * x[i];
      yy += static_cast<double>(y[i]) * y[i];
    }
    return d / std::sqrt(xx * yy);
  };
  const bool have_head = executor->layers() >= 45;
  const double cos_logits = have_head ? cosine(a, b) : 1.0;
  const double cos_res = cosine(ra, rb);
  const auto top = [](const std::vector<float>& l) {
    return std::max_element(l.begin(), l.end()) - l.begin();
  };
  const bool same_top = !have_head || top(a) == top(b);
  std::printf("tokens=%zu layers=%u chunk=%u\n", tokens.size(),
              executor->layers(), options.prefill_chunk);
  std::printf("sequential %.2f s (%.1f tok/s)\n", t_seq, tokens.size() / t_seq);
  std::printf("prefill    %.2f s (%.1f tok/s)\n", t_bat, tokens.size() / t_bat);
  std::printf("logits cos %.5f residual cos %.5f top1 %s\n", cos_logits,
              cos_res, same_top ? "same" : "DIFFERENT");
  const bool ok = cos_logits > kMinCosine && cos_res > kMinCosine && same_top;
  std::printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
