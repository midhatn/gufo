#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "ds4_mmq.h"
#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;

constexpr int kSkip = 77;
constexpr double kTolerance = 3e-2;
constexpr int kCheckRows = 256;
constexpr int kCheckCols = 6;

std::mt19937 g_rng(21);

std::vector<float> Rand(std::size_t n) {
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> v(n);
  for (auto& x : v)
    x = dist(g_rng);
  return v;
}

void DequantRow(GgmlType type, const char* src, float* dst, std::uint64_t k) {
  switch (type) {
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, dst, k);
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, dst, k);
      break;
    case GgmlType::kIQ2_XXS:
      gufo::quant::DequantizeIQ2_XXS(src, dst, k);
      break;
    case GgmlType::kIQ2_S:
      gufo::quant::DequantizeIQ2_S(src, dst, k);
      break;
    default:
      std::abort();
  }
}

void* DevCopy(const void* p, std::size_t n) {
  void* d = nullptr;
  hipMalloc(&d, n + 4096);
  hipMemcpy(d, p, n, hipMemcpyHostToDevice);
  return d;
}

// Relative error of selected (row, col) outputs of a column-major GEMM.
bool CheckDense(const GgufTensorInfo& t, int rows, int n, const char* what,
                int (*gemm)(const void*, const float*, float*, int, int, int,
                            hipStream_t)) {
  const int k = static_cast<int>(t.dimensions[0]);
  const std::size_t rb = t.size_bytes / t.dimensions[1];
  const auto x = Rand(static_cast<std::size_t>(n) * k);
  void* dw = DevCopy(t.data, rb * rows);
  float* dx = static_cast<float*>(DevCopy(x.data(), x.size() * 4));
  float* dout = nullptr;
  hipMalloc(&dout, static_cast<std::size_t>(rows) * n * 4);
  const int rc = gemm(dw, dx, dout, rows, n, k, nullptr);
  hipDeviceSynchronize();
  std::vector<float> got(static_cast<std::size_t>(rows) * n);
  hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost);
  double e2 = 0.0, r2 = 0.0;
  std::vector<float> w(k);
  for (int r = 0; r < std::min(rows, kCheckRows); ++r) {
    DequantRow(t.type, static_cast<const char*>(t.data) + r * rb, w.data(), k);
    for (int c = 0; c < n; c += std::max(1, n / kCheckCols)) {
      double acc = 0.0;
      for (int i = 0; i < k; ++i)
        acc += w[i] * x[static_cast<std::size_t>(c) * k + i];
      const double d = got[static_cast<std::size_t>(c) * rows + r] - acc;
      e2 += d * d;
      r2 += acc * acc;
    }
  }
  hipFree(dw);
  hipFree(dx);
  hipFree(dout);
  const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
  std::printf("%-26s M=%-5d N=%-4d K=%-5d rc=%d rel=%.2e %s\n", what, rows, n,
              k, rc, rel, rel < kTolerance && rc == 0 ? "OK" : "FAIL");
  return rel < kTolerance && rc == 0;
}

bool CheckMoe(const GgufTensorInfo& t, int tokens, const char* what,
              int (*moe)(const void*, const float*, const int32_t*, float*, int,
                         int, int, int, int, hipStream_t)) {
  const int k = static_cast<int>(t.dimensions[0]);
  const int m = static_cast<int>(t.dimensions[1]);
  const int experts = static_cast<int>(t.dimensions[2]);
  constexpr int kUsed = 8;
  const std::size_t rb = t.size_bytes / (static_cast<std::size_t>(m) * experts);
  const auto x = Rand(static_cast<std::size_t>(tokens) * k);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * kUsed);
  std::uniform_int_distribution<int> pick(0, experts - 1);
  for (int tk = 0; tk < tokens; ++tk) {
    for (int s = 0; s < kUsed; ++s) {
      int e;
      bool dup;
      do {
        e = pick(g_rng);
        dup = false;
        for (int q = 0; q < s; ++q)
          dup |= ids[tk * kUsed + q] == e;
      } while (dup);
      ids[tk * kUsed + s] = e;
    }
  }
  void* dw = DevCopy(t.data, t.size_bytes);
  float* dx = static_cast<float*>(DevCopy(x.data(), x.size() * 4));
  auto* dids = static_cast<std::int32_t*>(DevCopy(ids.data(), ids.size() * 4));
  float* dout = nullptr;
  hipMalloc(&dout, static_cast<std::size_t>(m) * tokens * kUsed * 4);
  const int rc = moe(dw, dx, dids, dout, m, k, tokens, experts, kUsed, nullptr);
  hipDeviceSynchronize();
  std::vector<float> got(static_cast<std::size_t>(m) * tokens * kUsed);
  hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost);
  double e2 = 0.0, r2 = 0.0;
  std::vector<float> w(k);
  for (int tk = 0; tk < tokens; tk += std::max(1, tokens / kCheckCols)) {
    for (int s = 0; s < kUsed; s += 3) {
      const int e = ids[tk * kUsed + s];
      for (int r = 0; r < kCheckRows; r += 3) {
        DequantRow(t.type,
                   static_cast<const char*>(t.data) +
                       (static_cast<std::size_t>(e) * m + r) * rb,
                   w.data(), k);
        double acc = 0.0;
        for (int i = 0; i < k; ++i)
          acc += w[i] * x[static_cast<std::size_t>(tk) * k + i];
        const double d =
            got[(static_cast<std::size_t>(tk) * kUsed + s) * m + r] - acc;
        e2 += d * d;
        r2 += acc * acc;
      }
    }
  }
  hipFree(dw);
  hipFree(dx);
  hipFree(dids);
  hipFree(dout);
  const double rel = std::sqrt(e2 / std::max(r2, 1e-30));
  std::printf("%-26s M=%-5d T=%-4d K=%-5d rc=%d rel=%.2e %s\n", what, m, tokens,
              k, rc, rel, rel < kTolerance && rc == 0 ? "OK" : "FAIL");
  return rel < kTolerance && rc == 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '\0') {
    std::printf("model artifact unavailable\n");
    return kSkip;
  }
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader) {
    std::printf("model artifact unavailable: %s\n", error.c_str());
    return kSkip;
  }
  ds4_mmq_init(0);
  const auto find = [&](const char* name) { return reader->FindTensor(name); };
  const auto* q6 = find("blk.0.attn_q.weight");
  const auto* q80 = find("blk.3.indexer.attn_q_b.weight");
  const auto* gate = find("blk.3.ffn_gate_exps.weight");
  const auto* down = find("blk.3.ffn_down_exps.weight");
  if (!q6 || !q80 || !gate || !down)
    return 1;
  bool ok = true;
  for (const int n : {1, 7, 64, 200}) {
    ok &= CheckDense(*q6, 2048, n, "q6_K dense", ds4_mmq_q6_K_dense);
    ok &= CheckDense(*q80, 1024, n, "q8_0 dense", ds4_mmq_q8_0_dense);
  }
  for (const int t : {1, 16, 100}) {
    ok &= CheckMoe(*gate, t, "iq2_xxs moe (gate)", ds4_mmq_iq2_xxs_moe);
    ok &= CheckMoe(*down, t,
                   down->type == GgmlType::kIQ2_S ? "iq2_s moe (down)"
                                                  : "iq2_xxs moe (down)",
                   down->type == GgmlType::kIQ2_S ? ds4_mmq_iq2_s_moe
                                                  : ds4_mmq_iq2_xxs_moe);
  }
  return ok ? 0 : 1;
}
