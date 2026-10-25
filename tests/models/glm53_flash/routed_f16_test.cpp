// Checks the Flash-Next F16 routed-expert GEMMs on GLM's IQ2 experts: the
// paired gate/up with clamped SwiGLU, then the down projection, against a
// double-precision CPU evaluation fed the same F16 inputs.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;
namespace qfn = gufo::models::qwen38_flash_next::rocm;

constexpr int kSkip = 77;
constexpr int kUsed = 8;
constexpr int kExperts = 288;
constexpr std::uint32_t kPadRows = 16;
constexpr double kTolerance = 2e-2;

qfn::WeightType Type(GgmlType t) {
  return t == GgmlType::kIQ2_S ? qfn::WeightType::kIQ2_S
                               : qfn::WeightType::kIQ2_XXS;
}

void Dequant(const GgufTensorInfo& t, std::uint64_t row, float* dst,
             std::uint64_t k) {
  const std::size_t rb = t.size_bytes / (t.dimensions[1] * t.dimensions[2]);
  const char* src = static_cast<const char*>(t.data) + row * rb;
  if (t.type == GgmlType::kIQ2_S)
    gufo::quant::DequantizeIQ2_S(src, dst, k);
  else
    gufo::quant::DequantizeIQ2_XXS(src, dst, k);
}

void* Dev(const GgufTensorInfo& t) {
  void* d = nullptr;
  hipMalloc(&d, t.size_bytes + 4096);
  hipMemcpy(d, t.data, t.size_bytes, hipMemcpyHostToDevice);
  return d;
}

double Sig(double v) {
  return 1.0 / (1.0 + std::exp(-v));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argv[1][0] == '\0')
    return kSkip;
  const int tokens = argc > 3 ? std::atoi(argv[3]) : 256;
  const std::string layer = argc > 2 ? argv[2] : "3";
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader)
    return kSkip;
  const auto* gate =
      reader->FindTensor("blk." + layer + ".ffn_gate_exps.weight");
  const auto* up = reader->FindTensor("blk." + layer + ".ffn_up_exps.weight");
  const auto* down =
      reader->FindTensor("blk." + layer + ".ffn_down_exps.weight");
  if (!gate || !up || !down)
    return 1;
  const int hidden = static_cast<int>(gate->dimensions[0]);
  const int ff = static_cast<int>(gate->dimensions[1]);
  const int slots = tokens * kUsed;

  std::mt19937 rng(17);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  std::vector<__half> x(static_cast<std::size_t>(tokens) * hidden);
  for (auto& v : x)
    v = __float2half(nd(rng));
  std::vector<std::int32_t> ids(slots);
  std::uniform_int_distribution<int> pick(0, kExperts - 1);
  for (int t = 0; t < tokens; ++t) {
    for (int s = 0; s < kUsed; ++s) {
      int e;
      bool dup;
      do {
        e = pick(rng);
        dup = false;
        for (int q = 0; q < s; ++q)
          dup |= ids[t * kUsed + q] == e;
      } while (dup);
      ids[t * kUsed + s] = e;
    }
  }
  void *dg = Dev(*gate), *du = Dev(*up), *dd = Dev(*down);
  __half *dx, *dact, *ddown;
  std::int32_t *dids, *dbounds, *dcur, *drt, *drs, *dtp, *dtd;
  std::uint32_t* dcounts;
  const std::size_t rows = qfn::RoutedCompactRows(slots, kExperts);
  const std::size_t tile_cap =
      (slots + kExperts * (kPadRows - 1)) / kPadRows + kExperts + 1;
  hipMalloc(&dx, x.size() * 2);
  hipMalloc(&dact, static_cast<std::size_t>(slots) * ff * 2);
  hipMalloc(&ddown, static_cast<std::size_t>(slots) * hidden * 2);
  hipMalloc(&dids, slots * 4);
  hipMalloc(&dcounts, kExperts * 4);
  hipMalloc(&dbounds, (kExperts + 1) * 4);
  hipMalloc(&dcur, kExperts * 4);
  hipMalloc(&drt, rows * 4);
  hipMalloc(&drs, rows * 4);
  hipMalloc(&dtp, tile_cap * 4);
  hipMalloc(&dtd, tile_cap * 4);
  hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(dids, ids.data(), slots * 4, hipMemcpyHostToDevice);

  const std::uint32_t pair_rows = 128;
  const std::uint32_t down_rows = argc > 4 ? std::atoi(argv[4]) : 64;
  qfn::ExpertCounts(dids, dcounts, tokens, kExperts, kUsed, nullptr);
  std::vector<std::uint32_t> counts(kExperts);
  hipMemcpy(counts.data(), dcounts, kExperts * 4, hipMemcpyDeviceToHost);
  std::vector<std::int32_t> tp, td;
  for (int e = 0; e < kExperts; ++e) {
    const std::uint32_t padded =
        (counts[e] + kPadRows - 1) / kPadRows * kPadRows;
    for (std::uint32_t j = 0; j < (padded + pair_rows - 1) / pair_rows; ++j)
      tp.push_back(e | (j << 16));
    for (std::uint32_t j = 0; j < (padded + down_rows - 1) / down_rows; ++j)
      td.push_back(e | (j << 16));
  }
  hipMemcpy(dtp, tp.data(), tp.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dtd, td.data(), td.size() * 4, hipMemcpyHostToDevice);
  qfn::RoutedCompact(dids, dcounts, dbounds, dcur, drt, drs, tokens, kUsed,
                     kExperts, nullptr);

  const auto wt_gate = Type(gate->type);
  const auto wt_down = Type(down->type);
  bool ok1 =
      qfn::RoutedGatedF16Gemm(dg, du, wt_gate, dx, dtp, tp.size(), pair_rows,
                              dbounds, drt, drs, dact, ff, hidden, nullptr);
  bool ok2 = qfn::RoutedF16Gemm(dd, wt_down, dact, dtd, td.size(), down_rows,
                                dbounds, drs, drs, nullptr, nullptr, ddown,
                                hidden, ff, nullptr);
  hipDeviceSynchronize();
  std::printf("launch gate/up=%d down=%d\n", ok1, ok2);
  if (!ok1 || !ok2)
    return 1;

  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  float ms_pair = 0, ms_down = 0;
  hipEventRecord(a);
  for (int i = 0; i < 5; ++i)
    qfn::RoutedGatedF16Gemm(dg, du, wt_gate, dx, dtp, tp.size(), pair_rows,
                            dbounds, drt, drs, dact, ff, hidden, nullptr);
  hipEventRecord(b);
  hipEventSynchronize(b);
  hipEventElapsedTime(&ms_pair, a, b);
  hipEventRecord(a);
  for (int i = 0; i < 5; ++i)
    qfn::RoutedF16Gemm(dd, wt_down, dact, dtd, td.size(), down_rows, dbounds,
                       drs, drs, nullptr, nullptr, ddown, hidden, ff, nullptr);
  hipEventRecord(b);
  hipEventSynchronize(b);
  hipEventElapsedTime(&ms_down, a, b);
  std::printf("T=%d gate+up %.2f ms  down %.2f ms  (per layer %.2f ms)\n",
              tokens, ms_pair / 5, ms_down / 5, ms_pair / 5 + ms_down / 5);

  std::vector<__half> act(static_cast<std::size_t>(slots) * ff),
      dn(static_cast<std::size_t>(slots) * hidden);
  hipMemcpy(act.data(), dact, act.size() * 2, hipMemcpyDeviceToHost);
  hipMemcpy(dn.data(), ddown, dn.size() * 2, hipMemcpyDeviceToHost);

  // CPU check on a sample of slots.
  double e2 = 0, r2 = 0, e3 = 0, r3 = 0;
  std::vector<float> wg(hidden), wu(hidden), wd(ff);
  for (int slot = 0; slot < slots; slot += std::max(1, slots / 24)) {
    const int t = slot / kUsed;
    const int e = ids[slot];
    std::vector<double> a_ref(ff);
    for (int r = 0; r < ff; r += 7) {
      Dequant(*gate, static_cast<std::uint64_t>(e) * ff + r, wg.data(), hidden);
      Dequant(*up, static_cast<std::uint64_t>(e) * ff + r, wu.data(), hidden);
      double g = 0, u = 0;
      for (int i = 0; i < hidden; ++i) {
        const double xv =
            __half2float(x[static_cast<std::size_t>(t) * hidden + i]);
        g += wg[i] * xv;
        u += wu[i] * xv;
      }
      g = std::min(g, 10.0);
      u = std::clamp(u, -10.0, 10.0);
      const double ref = g * Sig(g) * u;
      const double got =
          __half2float(act[static_cast<std::size_t>(slot) * ff + r]);
      e2 += (got - ref) * (got - ref);
      r2 += ref * ref;
    }
    for (int r = 0; r < hidden; r += 97) {
      Dequant(*down, static_cast<std::uint64_t>(e) * hidden + r, wd.data(), ff);
      double acc = 0;
      for (int i = 0; i < ff; ++i)
        acc +=
            wd[i] * __half2float(act[static_cast<std::size_t>(slot) * ff + i]);
      const double got =
          __half2float(dn[static_cast<std::size_t>(slot) * hidden + r]);
      e3 += (got - acc) * (got - acc);
      r3 += acc * acc;
    }
  }
  const double rel_act = std::sqrt(e2 / std::max(r2, 1e-30));
  const double rel_down = std::sqrt(e3 / std::max(r3, 1e-30));
  const bool ok = rel_act < kTolerance && rel_down < kTolerance;
  std::printf("act rel=%.3e  down rel=%.3e %s\n", rel_act, rel_down,
              ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}
