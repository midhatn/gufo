#include "src/models/glm53_flash/kernels/rocm/moe.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;
namespace rocm = gufo::models::glm53_flash::rocm;

constexpr int kSkip = 77;
constexpr int kHidden = 4096;
constexpr int kExpertFf = 2048;
constexpr int kDenseFf = 12288;
constexpr int kExperts = rocm::kMoeExperts;
constexpr int kUsed = rocm::kMoeUsed;
constexpr double kRouteScale = 2.5;
constexpr double kChainTolerance = 5e-2;

std::mt19937 g_rng(9);

std::vector<float> Rand(std::size_t n, float scale) {
  std::normal_distribution<float> dist(0.0F, scale);
  std::vector<float> v(n);
  for (auto& x : v)
    x = dist(g_rng);
  return v;
}

template<typename T>
T* Dev(const std::vector<T>& v) {
  T* d = nullptr;
  hipMalloc(&d, v.size() * sizeof(T));
  hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

template<typename T>
T* DevZero(std::size_t n) {
  T* d = nullptr;
  hipMalloc(&d, n * sizeof(T));
  hipMemset(d, 0, n * sizeof(T));
  return d;
}

template<typename T>
std::vector<T> Host(const T* d, std::size_t n) {
  std::vector<T> v(n);
  hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
  return v;
}

double Rel(const std::vector<float>& got, const std::vector<double>& ref) {
  double e = 0.0, r = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    e += (got[i] - ref[i]) * (got[i] - ref[i]);
    r += ref[i] * ref[i];
  }
  return std::sqrt(e / std::max(r, 1e-30));
}

double Sigmoid(double v) {
  return 1.0 / (1.0 + std::exp(-v));
}

double Swiglu(double gate, double up) {
  const double g = std::min(gate, 10.0);
  return g * Sigmoid(g) * std::clamp(up, -10.0, 10.0);
}

int Code(GgmlType t) {
  switch (t) {
    case GgmlType::kF32:
      return rocm::kTypeF32;
    case GgmlType::kQ8_0:
      return rocm::kTypeQ8_0;
    case GgmlType::kQ6_K:
      return rocm::kTypeQ6_K;
    case GgmlType::kIQ2_XXS:
      return rocm::kTypeIQ2_XXS;
    case GgmlType::kIQ2_S:
      return rocm::kTypeIQ2_S;
    default:
      return -1;
  }
}

void DequantRow(const GgufTensorInfo& t, const char* src, float* dst,
                std::uint64_t k) {
  switch (t.type) {
    case GgmlType::kF32:
      std::copy_n(reinterpret_cast<const float*>(src), k, dst);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, dst, k);
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

// y[row] for `nrows` rows starting at `row0` of one tensor (slab-aware).
std::vector<double> MatVec(const GgufTensorInfo& t, std::uint64_t row0,
                           std::uint64_t nrows, const std::vector<double>& x) {
  const std::uint64_t k = t.dimensions[0];
  std::uint64_t rows = 1;
  for (std::size_t d = 1; d < t.dimensions.size(); ++d)
    rows *= t.dimensions[d];
  const std::size_t rb = t.size_bytes / rows;
  std::vector<float> w(k);
  std::vector<double> y(nrows);
  for (std::uint64_t r = 0; r < nrows; ++r) {
    DequantRow(t, static_cast<const char*>(t.data) + (row0 + r) * rb, w.data(),
               k);
    double acc = 0.0;
    for (std::uint64_t i = 0; i < k; ++i)
      acc += w[i] * x[i];
    y[r] = acc;
  }
  return y;
}

void* DevTensor(const GgufTensorInfo& t) {
  void* d = nullptr;
  hipMalloc(&d, t.size_bytes);
  hipMemcpy(d, t.data, t.size_bytes, hipMemcpyHostToDevice);
  return d;
}

bool RouteCase(bool ties) {
  std::vector<float> logits = Rand(kExperts, 1.5F);
  std::vector<float> bias = Rand(kExperts, 0.2F);
  if (ties) {
    for (auto& v : logits)
      v = std::round(v * 2.0F) / 2.0F;
    std::fill(bias.begin(), bias.end(), 0.0F);
  }
  std::vector<double> prob(kExperts), sel(kExperts);
  for (int e = 0; e < kExperts; ++e) {
    prob[e] = static_cast<float>(1.0 / (1.0 + std::exp(-double(logits[e]))));
    sel[e] = static_cast<float>(prob[e] + bias[e]);
  }
  std::vector<int> order(kExperts);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return sel[a] > sel[b]; });
  double sum = 0.0;
  for (int i = 0; i < kUsed; ++i)
    sum += prob[order[i]];
  float *dl = Dev(logits), *db = Dev(bias);
  int* dids = DevZero<int>(kUsed);
  float* dw = DevZero<float>(kUsed);
  const int rc = rocm::MoeRoute(dl, db, static_cast<float>(kRouteScale), dids,
                                dw, nullptr);
  hipDeviceSynchronize();
  const auto ids = Host(dids, kUsed);
  const auto w = Host(dw, kUsed);
  bool ok = rc == 0;
  double werr = 0.0;
  for (int i = 0; i < kUsed; ++i) {
    ok &= ids[i] == order[i];
    werr = std::max(werr, std::fabs(w[i] - prob[order[i]] / sum * kRouteScale));
  }
  ok &= werr < 1e-5;
  std::printf("route ties=%d weights err=%.2e %s\n", ties, werr,
              ok ? "OK" : "FAIL");
  return ok;
}

bool MoeChain(const gufo::core::GgufReader& r, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto* inp = r.FindTensor(p + "ffn_gate_inp.weight");
  const auto* eb = r.FindTensor(p + "exp_probs_b.bias");
  const auto* ge = r.FindTensor(p + "ffn_gate_exps.weight");
  const auto* ue = r.FindTensor(p + "ffn_up_exps.weight");
  const auto* de = r.FindTensor(p + "ffn_down_exps.weight");
  const auto* gs = r.FindTensor(p + "ffn_gate_shexp.weight");
  const auto* us = r.FindTensor(p + "ffn_up_shexp.weight");
  const auto* ds = r.FindTensor(p + "ffn_down_shexp.weight");
  if (!inp || !eb || !ge || !ue || !de || !gs || !us || !ds) {
    std::printf("missing MoE tensors in layer %d\n", layer);
    return false;
  }
  void *d_inp = DevTensor(*inp), *d_ge = DevTensor(*ge), *d_ue = DevTensor(*ue),
       *d_de = DevTensor(*de), *d_gs = DevTensor(*gs), *d_us = DevTensor(*us),
       *d_ds = DevTensor(*ds);
  float* d_eb =
      Dev(std::vector<float>(static_cast<const float*>(eb->data),
                             static_cast<const float*>(eb->data) + kExperts));
  void* scratch = nullptr;
  hipMalloc(&scratch, 1 << 20);
  float *x = DevZero<float>(kHidden), *logits = DevZero<float>(kExperts),
        *gate = DevZero<float>(kUsed * kExpertFf),
        *up = DevZero<float>(kUsed * kExpertFf),
        *act = DevZero<float>(kUsed * kExpertFf),
        *parts = DevZero<float>(kUsed * kHidden),
        *sg = DevZero<float>(kExpertFf), *su = DevZero<float>(kExpertFf),
        *sact = DevZero<float>(kExpertFf), *shared = DevZero<float>(kHidden),
        *out = DevZero<float>(kHidden), *wts = DevZero<float>(kUsed);
  int* ids = DevZero<int>(kUsed);

  bool ok = true;
  double worst = 0.0;
  for (int trial = 0; trial < 3; ++trial) {
    const auto xin = Rand(kHidden, 1.0F);
    hipMemcpy(x, xin.data(), kHidden * 4, hipMemcpyHostToDevice);
    int rc = 0;
    rc |= rocm::Matvec(rocm::kTypeF32, d_inp, x, logits, kExperts, 1, kHidden,
                       nullptr, nullptr);
    rc |= rocm::MoeRoute(logits, d_eb, static_cast<float>(kRouteScale), ids,
                         wts, nullptr);
    rc |= rocm::MatvecExperts(Code(ge->type), d_ge, x, gate, kExpertFf, kHidden,
                              kExperts, kUsed, ids, false, scratch, nullptr);
    rc |= rocm::MatvecExperts(Code(ue->type), d_ue, x, up, kExpertFf, kHidden,
                              kExperts, kUsed, ids, false, scratch, nullptr);
    rc |= rocm::SwigluClamp(gate, up, act, kUsed * kExpertFf, nullptr);
    rc |= rocm::MatvecExperts(Code(de->type), d_de, act, parts, kHidden,
                              kExpertFf, kExperts, kUsed, ids, true, scratch,
                              nullptr);
    rc |= rocm::Matvec(Code(gs->type), d_gs, x, sg, kExpertFf, 1, kHidden,
                       scratch, nullptr);
    rc |= rocm::Matvec(Code(us->type), d_us, x, su, kExpertFf, 1, kHidden,
                       scratch, nullptr);
    rc |= rocm::SwigluClamp(sg, su, sact, kExpertFf, nullptr);
    rc |= rocm::Matvec(Code(ds->type), d_ds, sact, shared, kHidden, 1,
                       kExpertFf, scratch, nullptr);
    rc |= rocm::MoeCombine(parts, wts, shared, out, nullptr);
    hipDeviceSynchronize();

    const std::vector<double> xd(xin.begin(), xin.end());
    const auto lg = MatVec(*inp, 0, kExperts, xd);
    std::vector<double> prob(kExperts), selv(kExperts);
    for (int e = 0; e < kExperts; ++e) {
      prob[e] = Sigmoid(lg[e]);
      selv[e] = prob[e] + static_cast<const float*>(eb->data)[e];
    }
    std::vector<int> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return selv[a] > selv[b]; });
    const auto got_ids = Host(ids, kUsed);
    for (int i = 0; i < kUsed; ++i) {
      if (got_ids[i] != order[i]) {
        std::printf("layer %d route mismatch at %d: gpu=%d ref=%d\n", layer, i,
                    got_ids[i], order[i]);
        ok = false;
      }
    }
    double wsum = 0.0;
    for (int i = 0; i < kUsed; ++i)
      wsum += prob[order[i]];
    std::vector<double> ref(kHidden, 0.0);
    for (int i = 0; i < kUsed; ++i) {
      const int e = order[i];
      const auto g = MatVec(*ge, std::uint64_t(e) * kExpertFf, kExpertFf, xd);
      const auto u = MatVec(*ue, std::uint64_t(e) * kExpertFf, kExpertFf, xd);
      std::vector<double> a(kExpertFf);
      for (int j = 0; j < kExpertFf; ++j)
        a[j] = Swiglu(g[j], u[j]);
      const auto dp = MatVec(*de, std::uint64_t(e) * kHidden, kHidden, a);
      for (int c = 0; c < kHidden; ++c)
        ref[c] += prob[e] / wsum * kRouteScale * dp[c];
    }
    const auto g = MatVec(*gs, 0, kExpertFf, xd);
    const auto u = MatVec(*us, 0, kExpertFf, xd);
    std::vector<double> a(kExpertFf);
    for (int j = 0; j < kExpertFf; ++j)
      a[j] = Swiglu(g[j], u[j]);
    const auto sh = MatVec(*ds, 0, kHidden, a);
    for (int c = 0; c < kHidden; ++c)
      ref[c] += sh[c];
    const double e = Rel(Host(out, kHidden), ref);
    worst = std::max(worst, e);
    ok &= rc == 0 && e < kChainTolerance;
  }
  std::printf("layer %2d MoE (down type %d) 3 trials worst=%.2e %s\n", layer,
              Code(de->type), worst, ok ? "OK" : "FAIL");
  return ok;
}

bool DenseChain(const gufo::core::GgufReader& r, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto* g = r.FindTensor(p + "ffn_gate.weight");
  const auto* u = r.FindTensor(p + "ffn_up.weight");
  const auto* d = r.FindTensor(p + "ffn_down.weight");
  if (!g || !u || !d)
    return false;
  void *dg = DevTensor(*g), *du = DevTensor(*u), *dd = DevTensor(*d);
  void* scratch = nullptr;
  hipMalloc(&scratch, 1 << 20);
  float *x = DevZero<float>(kHidden), *gate = DevZero<float>(kDenseFf),
        *up = DevZero<float>(kDenseFf), *act = DevZero<float>(kDenseFf),
        *out = DevZero<float>(kHidden);
  // Large activations so the +-10 clamps actually engage.
  const auto xin = Rand(kHidden, 3.0F);
  hipMemcpy(x, xin.data(), kHidden * 4, hipMemcpyHostToDevice);
  int rc = rocm::Matvec(Code(g->type), dg, x, gate, kDenseFf, 1, kHidden,
                        scratch, nullptr);
  rc |= rocm::Matvec(Code(u->type), du, x, up, kDenseFf, 1, kHidden, scratch,
                     nullptr);
  rc |= rocm::SwigluClamp(gate, up, act, kDenseFf, nullptr);
  rc |= rocm::Matvec(Code(d->type), dd, act, out, kHidden, 1, kDenseFf, scratch,
                     nullptr);
  hipDeviceSynchronize();
  const std::vector<double> xd(xin.begin(), xin.end());
  const auto gr = MatVec(*g, 0, kDenseFf, xd);
  const auto ur = MatVec(*u, 0, kDenseFf, xd);
  std::vector<double> a(kDenseFf);
  int clamped = 0;
  for (int j = 0; j < kDenseFf; ++j) {
    a[j] = Swiglu(gr[j], ur[j]);
    clamped += (gr[j] > 10.0 || std::fabs(ur[j]) > 10.0);
  }
  const auto ref = MatVec(*d, 0, kHidden, a);
  const double e = Rel(Host(out, kHidden), ref);
  const bool ok = rc == 0 && e < kChainTolerance && clamped > 0;
  std::printf("layer %2d dense FFN clamped=%d err=%.2e %s\n", layer, clamped, e,
              ok ? "OK" : "FAIL");
  return ok;
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
  bool ok = RouteCase(false) & RouteCase(true);
  ok &= DenseChain(*reader, 0);
  ok &= MoeChain(*reader, 3);
  bool seen_xxs = false;
  for (int layer = 4; layer < 45 && !seen_xxs; ++layer) {
    const auto* de = reader->FindTensor("blk." + std::to_string(layer) +
                                        ".ffn_down_exps.weight");
    if (de != nullptr && de->type == GgmlType::kIQ2_XXS) {
      ok &= MoeChain(*reader, layer);
      seen_xxs = true;
    }
  }
  if (!seen_xxs)
    std::printf("note: no IQ2_XXS down layer in artifact\n");
  return ok ? 0 : 1;
}
