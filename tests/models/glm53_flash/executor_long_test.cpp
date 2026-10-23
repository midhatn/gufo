// Long teacher-forced comparison of the GPU executor against the CPU
// ReferenceModel over a token stream. Two invocations, so the two models are
// never resident together:
//   glm53_executor_long_test MODEL ref  TOKENS.txt DUMP.bin
//   glm53_executor_long_test MODEL gpu  TOKENS.txt DUMP.bin
// TOKENS.txt holds whitespace-separated token ids. GUFO_GLM53_TOPPOOLS caps
// the DSA selection window on both sides so short streams still exercise
// top-k selection.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/executor.hpp"
#include "src/models/glm53_flash/reference.hpp"

namespace {

constexpr int kSkip = 77;
constexpr int kTopK = 5;
constexpr double kMinCosine = 0.99;

std::vector<std::int32_t> ReadTokens(const char* path) {
  std::ifstream in(path);
  std::vector<std::int32_t> tokens;
  long v;
  while (in >> v)
    tokens.push_back(static_cast<std::int32_t>(v));
  return tokens;
}

std::vector<std::uint32_t> TopK(const std::vector<float>& l) {
  std::vector<std::uint32_t> order(l.size());
  for (std::uint32_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::partial_sort(
      order.begin(), order.begin() + kTopK, order.end(),
      [&](std::uint32_t a, std::uint32_t b) { return l[a] > l[b]; });
  order.resize(kTopK);
  return order;
}

double LogSumExp(const std::vector<float>& l) {
  float mx = *std::max_element(l.begin(), l.end());
  double s = 0.0;
  for (const float v : l)
    s += std::exp(static_cast<double>(v) - mx);
  return mx + std::log(s);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::printf("usage: %s MODEL ref|gpu TOKENS.txt DUMP.bin\n", argv[0]);
    return kSkip;
  }
  const std::string mode = argv[2];
  const auto tokens = ReadTokens(argv[3]);
  if (tokens.empty()) {
    std::printf("no tokens\n");
    return kSkip;
  }
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* pools_env = std::getenv("GUFO_GLM53_TOPPOOLS");
  const std::uint32_t top_pools =
      pools_env != nullptr ? static_cast<std::uint32_t>(std::atoi(pools_env))
                           : 512;
  std::string error;

  if (mode == "ref") {
    setenv("GUFO_GLM53_REF_TOPPOOLS", std::to_string(top_pools).c_str(), 1);
    auto model =
        gufo::models::glm53_flash::ReferenceModel::Open(argv[1], &error);
    if (!model) {
      std::printf("open failed: %s\n", error.c_str());
      return 1;
    }
    auto session = model->CreateSession();
    std::ofstream out(argv[4], std::ios::binary);
    for (std::size_t t = 0; t < tokens.size(); ++t) {
      const auto start = std::chrono::steady_clock::now();
      const auto& logits = model->Evaluate(*session, tokens[t]);
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
      out.flush();
      std::printf("ref t=%zu %.1f s\n", t,
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count());
    }
    return 0;
  }

  namespace rocm = gufo::models::glm53_flash::rocm;
  rocm::ExecutorOptions options;
  options.max_context = static_cast<std::uint32_t>(tokens.size()) + 8;
  options.top_pools = top_pools;
  auto executor = rocm::Executor::Create(argv[1], options, &error);
  if (!executor) {
    std::printf("create failed: %s\n", error.c_str());
    return 1;
  }
  auto session = executor->CreateSession(&error);
  if (!session) {
    std::printf("session failed: %s\n", error.c_str());
    return 1;
  }
  std::ifstream in(argv[4], std::ios::binary);
  int top1_match = 0;
  int bad_cosine = 0;
  double kl_sum = 0.0;
  double kl_max = 0.0;
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    if (!executor->Evaluate(*session, tokens[t], &error)) {
      std::printf("evaluate failed at %zu: %s\n", t, error.c_str());
      return 1;
    }
    const auto gpu = executor->Logits(*session);
    std::vector<float> ref(gpu.size());
    if (!in.read(reinterpret_cast<char*>(ref.data()),
                 static_cast<std::streamsize>(ref.size() * sizeof(float)))) {
      std::printf("dump too short at %zu\n", t);
      return kSkip;
    }
    double dot = 0.0, gg = 0.0, rr = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
      dot += static_cast<double>(gpu[i]) * ref[i];
      gg += static_cast<double>(gpu[i]) * gpu[i];
      rr += static_cast<double>(ref[i]) * ref[i];
    }
    const double cosine = dot / std::sqrt(gg * rr);
    const double lse_g = LogSumExp(gpu);
    const double lse_r = LogSumExp(ref);
    double kl = 0.0;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
      const double pr = std::exp(static_cast<double>(ref[i]) - lse_r);
      if (pr > 1e-12) {
        kl += pr * ((ref[i] - lse_r) - (gpu[i] - lse_g));
      }
    }
    const auto top_g = TopK(gpu);
    const auto top_r = TopK(ref);
    int overlap = 0;
    for (const auto id : top_r) {
      overlap += std::find(top_g.begin(), top_g.end(), id) != top_g.end();
    }
    top1_match += top_g[0] == top_r[0];
    bad_cosine += cosine < kMinCosine;
    kl_sum += kl;
    kl_max = std::max(kl_max, kl);
    std::printf("t=%3zu top1 %s overlap %d/5 cos %.5f KL %.4f\n", t,
                top_g[0] == top_r[0] ? "=" : "X", overlap, cosine, kl);
  }
  const double n = static_cast<double>(tokens.size());
  std::printf(
      "SUMMARY tokens=%zu top1=%d/%zu mean KL=%.4f max KL=%.4f "
      "cos<%.2f: %d (top_pools=%u)\n",
      tokens.size(), top1_match, tokens.size(), kl_sum / n, kl_max, kMinCosine,
      bad_cosine, top_pools);
  return bad_cosine == 0 ? 0 : 1;
}
