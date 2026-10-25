// Routed-expert matvec for a few tokens on real layer weights:
//   glm53_experts_bench MODEL [layer]
// Checks MatvecExpertsTokens against one MatvecExperts launch per token and
// reports the time and weight-stream rate of both.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

namespace {

namespace rocm = gufo::models::glm53_flash::rocm;
constexpr int kUsed = 8;
constexpr int kIters = 10;

int Code(gufo::core::GgmlType t) {
  switch (t) {
    case gufo::core::GgmlType::kIQ2_XXS:
      return rocm::kTypeIQ2_XXS;
    case gufo::core::GgmlType::kIQ2_S:
      return rocm::kTypeIQ2_S;
    default:
      return -1;
  }
}

float Ms(hipEvent_t a, hipEvent_t b) {
  float ms = 0;
  hipEventElapsedTime(&ms, a, b);
  return ms;
}

bool Bench(const gufo::core::GgufTensorInfo& t, int tokens, bool per_expert,
           const char* what) {
  const int k = static_cast<int>(t.dimensions[0]);
  const int m = static_cast<int>(t.dimensions[1]);
  const int experts = static_cast<int>(t.dimensions[2]);
  const int rows_x = per_expert ? tokens * kUsed : tokens;
  std::mt19937 rng(3);
  std::vector<float> x(static_cast<std::size_t>(rows_x) * k);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  for (auto& v : x)
    v = nd(rng);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * kUsed);
  std::uniform_int_distribution<int> pick(0, experts - 1);
  for (int i = 0; i < tokens; ++i) {
    for (int s = 0; s < kUsed; ++s) {
      int e;
      bool dup;
      do {
        e = pick(rng);
        dup = false;
        for (int q = 0; q < s; ++q)
          dup |= ids[i * kUsed + q] == e;
      } while (dup);
      ids[i * kUsed + s] = e;
    }
  }
  std::vector<int> distinct(ids.begin(), ids.end());
  std::sort(distinct.begin(), distinct.end());
  const int n_distinct = static_cast<int>(
      std::unique(distinct.begin(), distinct.end()) - distinct.begin());
  void* dw;
  hipMalloc(&dw, t.size_bytes + 4096);
  hipMemcpy(dw, t.data, t.size_bytes, hipMemcpyHostToDevice);
  float *dx, *dout, *dref;
  int* dids;
  void* scratch;
  const std::size_t out_floats = static_cast<std::size_t>(m) * tokens * kUsed;
  hipMalloc(&dx, x.size() * 4);
  hipMalloc(&dout, out_floats * 4);
  hipMalloc(&dref, out_floats * 4);
  hipMalloc(&dids, ids.size() * 4);
  hipMalloc(&scratch, (1 << 20));
  hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice);
  const int code = Code(t.type);

  const auto per_token = [&] {
    for (int i = 0; i < tokens; ++i) {
      rocm::MatvecExperts(
          code, dw,
          dx + static_cast<std::size_t>(i) * k * (per_expert ? kUsed : 1),
          dref + static_cast<std::size_t>(i) * kUsed * m, m, k, experts, kUsed,
          dids + i * kUsed, per_expert, scratch, nullptr);
    }
  };
  const auto batched = [&] {
    return rocm::MatvecExpertsTokens(code, dw, dx, dout, m, k, experts, kUsed,
                                     tokens, dids, per_expert, scratch,
                                     nullptr);
  };
  per_token();
  const int rc = batched();
  hipDeviceSynchronize();
  std::vector<float> a(out_floats), b(out_floats);
  hipMemcpy(a.data(), dout, out_floats * 4, hipMemcpyDeviceToHost);
  hipMemcpy(b.data(), dref, out_floats * 4, hipMemcpyDeviceToHost);
  double diff = 0, norm = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diff += std::fabs(a[i] - b[i]);
    norm += std::fabs(b[i]);
  }
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  hipEventRecord(e0);
  for (int i = 0; i < kIters; ++i)
    per_token();
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  const float ms_tok = Ms(e0, e1) / kIters;
  hipEventRecord(e0);
  for (int i = 0; i < kIters; ++i)
    batched();
  hipEventRecord(e1);
  hipEventSynchronize(e1);
  const float ms_bat = Ms(e0, e1) / kIters;
  const double bytes = static_cast<double>(t.size_bytes) / experts * n_distinct;
  std::printf(
      "%-5s T=%d distinct=%-3d rc=%d rel diff %.2e  per-token %.3f ms (%.0f "
      "GB/s)  batched %.3f ms (%.0f GB/s)\n",
      what, tokens, n_distinct, rc, diff / std::max(norm, 1e-30), ms_tok,
      bytes / ms_tok / 1e6, ms_bat, bytes / ms_bat / 1e6);
  hipFree(dw);
  hipFree(dx);
  hipFree(dout);
  hipFree(dref);
  hipFree(dids);
  hipFree(scratch);
  return rc == 0 && diff <= 1e-6 * std::max(norm, 1e-30);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '\0')
    return 77;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader)
    return 77;
  const std::string layer = argc > 2 ? argv[2] : "3";
  const auto* gate =
      reader->FindTensor("blk." + layer + ".ffn_gate_exps.weight");
  const auto* down =
      reader->FindTensor("blk." + layer + ".ffn_down_exps.weight");
  if (!gate || !down)
    return 1;
  bool ok = true;
  for (const int tokens : {1, 2, 4, 7, 8}) {
    ok &= Bench(*gate, tokens, false, "gate");
    ok &= Bench(*down, tokens, true, "down");
  }
  std::printf(ok ? "OK\n" : "FAIL\n");
  return ok ? 0 : 1;
}
