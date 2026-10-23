// Compares the GPU executor against the CPU ReferenceModel on a truncated
// trunk: after each teacher-forced token the 4x4096 residual of the first N
// layers must agree. N comes from GUFO_GLM53_TEST_LAYERS (default 4); the
// reference runs first and is released before the GPU weights are uploaded.
#include "src/models/glm53_flash/kernels/rocm/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/models/glm53_flash/reference.hpp"

namespace {

constexpr int kSkip = 77;
constexpr std::int32_t kTokens[] = {9703, 11, 1879, 0, 101726, 98384};
constexpr double kTolerance = 5e-2;

double Rel(const std::vector<float>& got, const std::vector<float>& ref) {
  double e = 0.0, r = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - ref[i];
    e += d * d;
    r += static_cast<double>(ref[i]) * ref[i];
  }
  return std::sqrt(e / std::max(r, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '\0') {
    std::printf("model artifact unavailable\n");
    return kSkip;
  }
  const char* env = std::getenv("GUFO_GLM53_TEST_LAYERS");
  const int layers = env != nullptr ? std::atoi(env) : 4;
  setenv("GUFO_GLM53_REF_LAYERS", std::to_string(layers).c_str(), 1);

  std::vector<std::vector<float>> ref_x;
  {
    std::string error;
    auto ref = gufo::models::glm53_flash::ReferenceModel::Open(argv[1], &error);
    if (!ref) {
      std::printf("model artifact unavailable: %s\n", error.c_str());
      return kSkip;
    }
    auto session = ref->CreateSession();
    for (const auto token : kTokens) {
      ref->Evaluate(*session, token);
      ref_x.push_back(session->x);
    }
  }

  namespace rocm = gufo::models::glm53_flash::rocm;
  rocm::ExecutorOptions options;
  options.layers = static_cast<std::uint32_t>(layers);
  options.head = false;
  options.max_context = 64;
  std::string error;
  auto executor = rocm::Executor::Create(argv[1], options, &error);
  if (!executor) {
    std::printf("executor create failed: %s\n", error.c_str());
    return 1;
  }
  auto session = executor->CreateSession(&error);
  if (!session) {
    std::printf("session create failed: %s\n", error.c_str());
    return 1;
  }
  bool ok = true;
  for (std::size_t t = 0; t < std::size(kTokens); ++t) {
    if (!executor->Evaluate(*session, kTokens[t], &error)) {
      std::printf("evaluate failed: %s\n", error.c_str());
      return 1;
    }
    const double e = Rel(executor->Residual(*session), ref_x[t]);
    ok &= e < kTolerance;
    std::printf("layers=%d token %zu residual rel=%.3e %s\n", layers, t, e,
                e < kTolerance ? "OK" : "FAIL");
  }
  return ok ? 0 : 1;
}
