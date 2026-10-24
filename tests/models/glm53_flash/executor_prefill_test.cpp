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
// near-tied expert routing through 45 layers (8 layers agree at 0.9997, and
// two chunkings of the same prompt differ from each other by as much as from
// sequential decode). The final logits are the contract; the residual cosine
// is only reported.
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

  const bool skip_seq = std::getenv("GUFO_GLM53_SKIP_SEQ") != nullptr;
  auto seq = executor->CreateSession(&error);
  auto start = std::chrono::steady_clock::now();
  for (const auto token : tokens) {
    if (skip_seq)
      break;
    if (!executor->Evaluate(*seq, token, &error)) {
      std::printf("evaluate failed: %s\n", error.c_str());
      return 1;
    }
  }
  const double t_seq = Seconds(start);
  std::printf("sequential stages:\n");
  executor->PrintProfile();
  const auto a = executor->Logits(*seq);
  const auto ra = executor->Residual(*seq);

  auto bat = executor->CreateSession(&error);
  start = std::chrono::steady_clock::now();
  std::size_t split = 0;
  if (const char* v = std::getenv("GUFO_GLM53_SPLIT"))
    split = std::atoi(v);
  if (!executor->Prefill(*bat, tokens, &error, split)) {
    std::printf("prefill failed: %s\n", error.c_str());
    return 1;
  }
  double t_bat = Seconds(start);
  if (const char* v = std::getenv("GUFO_GLM53_REPEAT")) {
    for (int i = 0; i < std::atoi(v); ++i) {
      auto again = executor->CreateSession(&error);
      const auto t = std::chrono::steady_clock::now();
      if (!executor->Prefill(*again, tokens, &error, split))
        return 1;
      t_bat = Seconds(t);
      std::printf("repeat %d: %.2f s (%.1f tok/s)\n", i, t_bat,
                  tokens.size() / t_bat);
    }
  }
  std::printf("prefill stages:\n");
  executor->PrintProfile();
  const auto b = executor->Logits(*bat);
  const auto rb = executor->Residual(*bat);

  const auto cosine_fn = [](const std::vector<float>& x,
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
  if (const char* path = std::getenv("GUFO_GLM53_DUMP")) {
    if (FILE* f = std::fopen(path, "wb")) {
      std::fwrite(b.data(), sizeof(float), b.size(), f);
      std::fclose(f);
    }
  }
  if (const char* path = std::getenv("GUFO_GLM53_CMP")) {
    std::vector<float> other(b.size());
    if (FILE* f = std::fopen(path, "rb")) {
      if (std::fread(other.data(), sizeof(float), other.size(), f) ==
          other.size()) {
        std::printf(
            "vs file %s: logits cos %.5f top1 %s\n", path, cosine_fn(b, other),
            std::max_element(b.begin(), b.end()) - b.begin() ==
                    std::max_element(other.begin(), other.end()) - other.begin()
                ? "same"
                : "DIFFERENT");
      }
      std::fclose(f);
    }
  }
  if (const char* v = std::getenv("GUFO_GLM53_REF_CHUNK")) {
    auto ref = executor->CreateSession(&error);
    if (!executor->Prefill(*ref, tokens, &error, std::atoi(v)))
      return 1;
    const auto lr = executor->Logits(*ref);
    const auto rr = executor->Residual(*ref);
    std::printf(
        "vs chunk %s prefill: logits cos %.5f residual cos %.5f top1 %s\n", v,
        cosine_fn(b, lr), cosine_fn(rb, rr),
        std::max_element(b.begin(), b.end()) - b.begin() ==
                std::max_element(lr.begin(), lr.end()) - lr.begin()
            ? "same"
            : "DIFFERENT");
  }
  if (skip_seq) {
    std::printf("prefill %.2f s (%.1f tok/s)\n", t_bat, tokens.size() / t_bat);
    return 0;
  }
  const auto cosine = cosine_fn;
  if (split > 0 && have_head) {
    auto whole = executor->CreateSession(&error);
    if (!executor->Prefill(*whole, tokens, &error, 0))
      return 1;
    const auto c = executor->Logits(*whole);
    std::printf(
        "chunked(%zu) vs single-chunk prefill: logits cos %.5f top1 %s\n",
        split, cosine(b, c),
        std::max_element(b.begin(), b.end()) - b.begin() ==
                std::max_element(c.begin(), c.end()) - c.begin()
            ? "same"
            : "DIFFERENT");
  }
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
  const bool ok = cos_logits > kMinCosine && same_top;
  std::printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
