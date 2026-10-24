#include "src/models/glm53_flash/kernels/rocm/dsa.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
constexpr int kLat = rocm::kDsaLatent;
constexpr int kHeads = rocm::kDsaHeads;
constexpr int kIh = rocm::kDsaIndexHeads;
constexpr int kId = rocm::kDsaIndexDim;
constexpr int kPool = rocm::kDsaPool;
constexpr int kQk = 256;
constexpr int kQlora = 1536;
constexpr int kHidden = 4096;
constexpr double kStageTolerance = 1e-5;
constexpr double kChainTolerance = 5e-2;

std::mt19937 g_rng(5);

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

float RoundHalf(float f) {
  return static_cast<float>(static_cast<_Float16>(f));
}

std::uint16_t* DevHalf(const std::vector<float>& v) {
  std::vector<std::uint16_t> bits(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    const _Float16 h = static_cast<_Float16>(v[i]);
    std::memcpy(&bits[i], &h, sizeof(std::uint16_t));
  }
  std::uint16_t* d = nullptr;
  hipMalloc(&d, bits.size() * sizeof(std::uint16_t));
  hipMemcpy(d, bits.data(), bits.size() * sizeof(std::uint16_t),
            hipMemcpyHostToDevice);
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

bool Report(const char* what, double err, double tol) {
  const bool ok = err < tol;
  std::printf("%-44s err=%.2e %s\n", what, err, ok ? "OK" : "FAIL");
  return ok;
}

bool SelectCase(int npool, int top, bool ties) {
  std::vector<float> scores = Rand(npool, 3.0F);
  if (ties) {
    for (auto& s : scores)
      s = std::round(s) / 2.0F + 0.0F;
  }
  std::vector<int> order(npool);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return scores[a] > scores[b]; });
  const int k = std::min(npool, top);
  std::vector<int> want(order.begin(), order.begin() + k);
  std::sort(want.begin(), want.end());
  float* ds = Dev(scores);
  int* dsel = DevZero<int>(npool + 1);
  const int rc = rocm::DsaSelect(ds, npool, top, dsel, nullptr);
  hipDeviceSynchronize();
  const auto got = Host(dsel, k);
  hipFree(ds);
  hipFree(dsel);
  const bool ok = rc == 0 && got == want;
  std::printf("select npool=%-6d top=%-4d ties=%d %s\n", npool, top, ties,
              ok ? "OK" : "FAIL");
  return ok;
}

bool StageTests(const gufo::core::GgufReader& reader) {
  bool ok = true;
  for (const bool ties : {false, true}) {
    ok &= SelectCase(3000, 512, ties);
    ok &= SelectCase(32768, 512, ties);
    ok &= SelectCase(513, 512, ties);
    ok &= SelectCase(512, 512, ties);
    ok &= SelectCase(300, 512, ties);
    ok &= SelectCase(48, 8, ties);
  }

  {  // Scores
    const int npool = 1001;
    const auto iq = Rand(kIh * kId, 1.0F);
    const auto proj = Rand(kIh, 1.0F);
    const auto pooled = Rand(static_cast<std::size_t>(npool) * kId, 1.0F);
    std::vector<double> ref(npool);
    for (int p = 0; p < npool; ++p) {
      double acc = 0.0;
      for (int h = 0; h < kIh; ++h) {
        double dot = 0.0;
        for (int d = 0; d < kId; ++d)
          dot += iq[h * kId + d] * pooled[p * kId + d];
        acc += std::max(dot, 0.0) * proj[h] / std::sqrt(double(kId) * kIh);
      }
      ref[p] = acc;
    }
    float *diq = Dev(iq), *dproj = Dev(proj), *dpool = Dev(pooled);
    float* dscore = DevZero<float>(npool);
    const int rc = rocm::DsaScores(diq, dproj, dpool, npool, dscore, nullptr);
    hipDeviceSynchronize();
    ok &= rc == 0 &&
          Report("scores", Rel(Host(dscore, npool), ref), kStageTolerance);
    for (float* p : {diq, dproj, dpool, dscore})
      hipFree(p);
  }

  {  // Attend
    const int n = 2302;
    const int pos = n - 1;
    const int tail = (pos + 1) % kPool;
    const int npool = (pos + 1) / kPool;
    auto lat = Rand(static_cast<std::size_t>(n) * kLat, 1.0F);
    for (auto& x : lat)
      x = RoundHalf(x);
    const auto qabs = Rand(kHeads * kLat, 0.3F);
    std::vector<int> pools(npool);
    std::iota(pools.begin(), pools.end(), 0);
    std::shuffle(pools.begin(), pools.end(), g_rng);
    pools.resize(rocm::kDsaTopPools);
    std::sort(pools.begin(), pools.end());
    std::vector<int> vis;
    for (const int p : pools)
      for (int s = 0; s < kPool; ++s)
        vis.push_back(p * kPool + s);
    for (int k = 0; k < tail; ++k)
      vis.push_back(pos - k);
    const double scale = 1.0 / 16.0;
    std::vector<double> ref(kHeads * kLat);
    for (int h = 0; h < kHeads; ++h) {
      std::vector<double> sc(vis.size());
      double mx = -1e300;
      for (std::size_t j = 0; j < vis.size(); ++j) {
        double dot = 0.0;
        for (int c = 0; c < kLat; ++c)
          dot += qabs[h * kLat + c] *
                 lat[static_cast<std::size_t>(vis[j]) * kLat + c];
        sc[j] = dot * scale;
        mx = std::max(mx, sc[j]);
      }
      double den = 0.0;
      for (auto& s : sc)
        den += (s = std::exp(s - mx));
      for (std::size_t j = 0; j < vis.size(); ++j)
        for (int c = 0; c < kLat; ++c)
          ref[h * kLat + c] +=
              sc[j] / den * lat[static_cast<std::size_t>(vis[j]) * kLat + c];
    }
    std::uint16_t* dlat = DevHalf(lat);
    float* dq = Dev(qabs);
    int* dsel = Dev(pools);
    float* dctx = DevZero<float>(kHeads * kLat);
    const int rc = rocm::DsaAttend(dq, dlat, dsel, rocm::kDsaTopPools, pos,
                                   static_cast<float>(scale), dctx, nullptr);
    hipDeviceSynchronize();
    ok &=
        rc == 0 && Report("attend (512 pools + tail 2)",
                          Rel(Host(dctx, kHeads * kLat), ref), kStageTolerance);
    for (void* p : {static_cast<void*>(dlat), static_cast<void*>(dq),
                    static_cast<void*>(dsel), static_cast<void*>(dctx)})
      hipFree(p);
  }

  {  // Append + pooling with real ape
    const auto* ape_t =
        reader.FindTensor("blk.3.indexer_compressor_ape.weight");
    if (ape_t == nullptr)
      return false;
    const float* ape = static_cast<const float*>(ape_t->data);
    const int tokens = 12;
    const auto lat = Rand(tokens * kLat, 1.0F);
    const auto ik = Rand(tokens * kId, 1.0F);
    const auto gate = Rand(tokens * kId, 1.0F);
    rocm::DsaCache cache{DevZero<std::uint16_t>(tokens * kLat),
                         DevZero<float>(8 * kId), DevZero<float>(8 * kId),
                         DevZero<float>((tokens / kPool) * kId), 8};
    float* dape = Dev(std::vector<float>(ape, ape + kPool * kId));
    float *dlat = Dev(lat), *dik = Dev(ik), *dgate = Dev(gate);
    int rc = 0;
    for (int t = 0; t < tokens; ++t) {
      rc |= rocm::DsaAppend(cache, dlat + t * kLat, dik + t * kId,
                            dgate + t * kId, dape, t, nullptr);
    }
    hipDeviceSynchronize();
    std::vector<double> ref;
    for (int pidx = 0; pidx < tokens / kPool; ++pidx) {
      for (int d = 0; d < kId; ++d) {
        double x4[kPool], mx = -1e300, sum = 0.0, acc = 0.0;
        for (int s = 0; s < kPool; ++s) {
          x4[s] = gate[(pidx * kPool + s) * kId + d] + ape[d + kId * s];
          mx = std::max(mx, x4[s]);
        }
        for (int s = 0; s < kPool; ++s)
          sum += (x4[s] = std::exp(x4[s] - mx));
        for (int s = 0; s < kPool; ++s)
          acc += x4[s] / sum * ik[(pidx * kPool + s) * kId + d];
        ref.push_back(acc);
      }
    }
    ok &= rc == 0 &&
          Report("append + pooling", Rel(Host(cache.pooled, ref.size()), ref),
                 kStageTolerance);
  }
  return ok;
}

// ---- Full-layer chain on real weights ----

struct Mat {
  const GgufTensorInfo* t{nullptr};
  std::vector<float> w;  // dequantized, [slab][row][k]
  std::uint64_t k{0}, rows{0};
  int code{-1};
};

int Code(GgmlType t) {
  switch (t) {
    case GgmlType::kF32:
      return rocm::kTypeF32;
    case GgmlType::kQ8_0:
      return rocm::kTypeQ8_0;
    case GgmlType::kQ6_K:
      return rocm::kTypeQ6_K;
    default:
      return -1;
  }
}

Mat LoadMat(const gufo::core::GgufReader& r, const std::string& name) {
  Mat m;
  m.t = r.FindTensor(name);
  if (m.t == nullptr)
    return m;
  m.k = m.t->dimensions[0];
  m.rows = 1;
  for (std::size_t d = 1; d < m.t->dimensions.size(); ++d)
    m.rows *= m.t->dimensions[d];
  m.code = Code(m.t->type);
  m.w.resize(m.rows * m.k);
  const std::size_t rb = m.t->size_bytes / m.rows;
  for (std::uint64_t row = 0; row < m.rows; ++row) {
    const char* src = static_cast<const char*>(m.t->data) + row * rb;
    float* dst = m.w.data() + row * m.k;
    switch (m.t->type) {
      case GgmlType::kF32:
        std::copy_n(reinterpret_cast<const float*>(src), m.k, dst);
        break;
      case GgmlType::kQ8_0:
        gufo::quant::DequantizeQ8_0(src, dst, m.k);
        break;
      case GgmlType::kQ6_K:
        gufo::quant::DequantizeQ6_K(src, dst, m.k);
        break;
      default:
        std::abort();
    }
  }
  return m;
}

std::vector<double> MatVec(const Mat& m, const std::vector<double>& x,
                           std::uint64_t row0 = 0, std::uint64_t nrows = 0) {
  if (nrows == 0)
    nrows = m.rows;
  std::vector<double> y(nrows);
  for (std::uint64_t r = 0; r < nrows; ++r) {
    const float* w = m.w.data() + (row0 + r) * m.k;
    double acc = 0.0;
    for (std::uint64_t i = 0; i < m.k; ++i)
      acc += w[i] * x[i];
    y[r] = acc;
  }
  return y;
}

void RmsD(std::vector<double>& x, const float* w, double eps) {
  double ss = 0.0;
  for (const double v : x)
    ss += v * v;
  const double inv = 1.0 / std::sqrt(ss / x.size() + eps);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] *= inv * w[i];
}

std::vector<double> ToD(const std::vector<float>& v) {
  return {v.begin(), v.end()};
}

struct Layer {
  Mat qa, qb, kva, iqb, ik, gate, proj, kb, vb, out;
  const float *qa_norm, *kva_norm, *knw, *knb, *ape;
};

bool ChainTest(const gufo::core::GgufReader& reader, int layer_idx) {
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  Layer L;
  L.qa = LoadMat(reader, p + "attn_q_a.weight");
  L.qb = LoadMat(reader, p + "attn_q_b.weight");
  L.kva = LoadMat(reader, p + "attn_kv_a_mqa.weight");
  L.iqb = LoadMat(reader, p + "indexer.attn_q_b.weight");
  L.ik = LoadMat(reader, p + "indexer.attn_k.weight");
  L.gate = LoadMat(reader, p + "indexer_compressor_gate.weight");
  L.proj = LoadMat(reader, p + "indexer.proj.weight");
  L.kb = LoadMat(reader, p + "attn_k_b.weight");
  L.vb = LoadMat(reader, p + "attn_v_b.weight");
  L.out = LoadMat(reader, p + "attn_output.weight");
  const auto f32 = [&](const char* n) {
    const auto* t = reader.FindTensor(p + n);
    return t ? static_cast<const float*>(t->data) : nullptr;
  };
  L.qa_norm = f32("attn_q_a_norm.weight");
  L.kva_norm = f32("attn_kv_a_norm.weight");
  L.knw = f32("indexer.k_norm.weight");
  L.knb = f32("indexer.k_norm.bias");
  L.ape = f32("indexer_compressor_ape.weight");
  if (!L.qa.t || !L.qb.t || !L.kva.t || !L.iqb.t || !L.ik.t || !L.gate.t ||
      !L.proj.t || !L.kb.t || !L.vb.t || !L.out.t || !L.qa_norm ||
      !L.kva_norm || !L.knw || !L.knb || !L.ape) {
    std::printf("missing DSA tensors in layer %d\n", layer_idx);
    return false;
  }

  constexpr int kTokens = 40;
  constexpr int kTop = 6;
  const auto dev_tensor = [&](const Mat& m) {
    void* d = nullptr;
    hipMalloc(&d, m.t->size_bytes);
    hipMemcpy(d, m.t->data, m.t->size_bytes, hipMemcpyHostToDevice);
    return d;
  };
  void *d_qa = dev_tensor(L.qa), *d_qb = dev_tensor(L.qb),
       *d_kva = dev_tensor(L.kva), *d_iqb = dev_tensor(L.iqb),
       *d_ik = dev_tensor(L.ik), *d_gate = dev_tensor(L.gate),
       *d_proj = dev_tensor(L.proj), *d_kb = dev_tensor(L.kb),
       *d_vb = dev_tensor(L.vb), *d_out = dev_tensor(L.out);
  const auto dvec = [&](const float* p_, std::size_t n) {
    return Dev(std::vector<float>(p_, p_ + n));
  };
  float *d_qa_norm = dvec(L.qa_norm, kQlora),
        *d_kva_norm = dvec(L.kva_norm, kLat), *d_knw = dvec(L.knw, kId),
        *d_knb = dvec(L.knb, kId), *d_ape = dvec(L.ape, kPool * kId);
  void* scratch = nullptr;
  hipMalloc(&scratch, 1 << 20);
  float *x = DevZero<float>(kHidden), *qr = DevZero<float>(kQlora),
        *q = DevZero<float>(kHeads * kQk), *lat = DevZero<float>(kLat),
        *iq = DevZero<float>(kIh * kId), *ik = DevZero<float>(kId),
        *gt = DevZero<float>(kId), *wp = DevZero<float>(kIh),
        *scores = DevZero<float>(kTokens),
        *qabs = DevZero<float>(kHeads * kLat),
        *ctx = DevZero<float>(kHeads * kLat),
        *vout = DevZero<float>(kHeads * kQk), *block = DevZero<float>(kHidden);
  int* sel = DevZero<int>(kTokens);
  rocm::DsaCache cache{DevZero<std::uint16_t>(kTokens * kLat),
                       DevZero<float>(8 * kId), DevZero<float>(8 * kId),
                       DevZero<float>((kTokens / kPool) * kId), 8};

  // Oracle state.
  std::vector<std::vector<double>> o_lat, o_ik, o_gate, o_pooled;
  bool ok = true;
  double worst = 0.0;
  for (int pos = 0; pos < kTokens; ++pos) {
    const auto xin = Rand(kHidden, 1.0F);
    hipMemcpy(x, xin.data(), kHidden * 4, hipMemcpyHostToDevice);
    int rc = 0;
    rc |= rocm::Matvec(L.qa.code, d_qa, x, qr, kQlora, 1, kHidden, scratch,
                       nullptr);
    rc |= rocm::RmsNorm(qr, d_qa_norm, kQlora, 1e-5F, nullptr);
    rc |= rocm::Matvec(L.qb.code, d_qb, qr, q, kHeads * kQk, 1, kQlora, scratch,
                       nullptr);
    rc |= rocm::Matvec(L.kva.code, d_kva, x, lat, kLat, 1, kHidden, scratch,
                       nullptr);
    rc |= rocm::RmsNorm(lat, d_kva_norm, kLat, 1e-5F, nullptr);
    rc |= rocm::Matvec(L.iqb.code, d_iqb, qr, iq, kIh * kId, 1, kQlora, scratch,
                       nullptr);
    rc |=
        rocm::Matvec(L.ik.code, d_ik, x, ik, kId, 1, kHidden, scratch, nullptr);
    rc |= rocm::LayerNorm(ik, d_knw, d_knb, kId, 1e-6F, nullptr);
    rc |= rocm::Matvec(L.gate.code, d_gate, x, gt, kId, 1, kHidden, scratch,
                       nullptr);
    rc |= rocm::DsaAppend(cache, lat, ik, gt, d_ape, pos, nullptr);
    rc |= rocm::Matvec(L.proj.code, d_proj, x, wp, kIh, 1, kHidden, scratch,
                       nullptr);
    const int npool = (pos + 1) / kPool;
    rc |= rocm::DsaScores(iq, wp, cache.pooled, npool, scores, nullptr);
    rc |= rocm::DsaSelect(scores, npool, kTop, sel, nullptr);
    rc |= rocm::MatvecBatched(L.kb.code, d_kb, q, qabs, kLat, kQk, kHeads,
                              scratch, nullptr);
    const int nsel = std::min(npool, kTop);
    rc |= rocm::DsaAttend(qabs, cache.lat_half, sel, nsel, pos,
                          1.0F / std::sqrt(float(kQk)), ctx, nullptr);
    rc |= rocm::MatvecBatched(L.vb.code, d_vb, ctx, vout, kQk, kLat, kHeads,
                              scratch, nullptr);
    rc |= rocm::Matvec(L.out.code, d_out, vout, block, kHidden, 1, kHeads * kQk,
                       scratch, nullptr);
    hipDeviceSynchronize();

    // Oracle.
    const auto xd = ToD(xin);
    auto qrd = MatVec(L.qa, xd);
    RmsD(qrd, L.qa_norm, 1e-5);
    const auto qd = MatVec(L.qb, qrd);
    auto latd = MatVec(L.kva, xd);
    RmsD(latd, L.kva_norm, 1e-5);
    const auto iqd = MatVec(L.iqb, qrd);
    auto ikd = MatVec(L.ik, xd);
    {
      double mu = 0.0, s2 = 0.0;
      for (const double v : ikd)
        mu += v;
      mu /= kId;
      for (const double v : ikd)
        s2 += (v - mu) * (v - mu);
      const double inv = 1.0 / std::sqrt(s2 / kId + 1e-6);
      for (int i = 0; i < kId; ++i)
        ikd[i] = (ikd[i] - mu) * inv * L.knw[i] + L.knb[i];
    }
    const auto gd = MatVec(L.gate, xd);
    for (auto& x : latd)
      x = RoundHalf(static_cast<float>(x));
    o_lat.push_back(latd);
    o_ik.push_back(ikd);
    o_gate.push_back(gd);
    if ((pos + 1) % kPool == 0) {
      const int b = pos + 1 - kPool;
      std::vector<double> pn(kId);
      for (int d = 0; d < kId; ++d) {
        double x4[kPool], mx = -1e300, sum = 0.0, acc = 0.0;
        for (int s = 0; s < kPool; ++s) {
          x4[s] = o_gate[b + s][d] + L.ape[d + kId * s];
          mx = std::max(mx, x4[s]);
        }
        for (int s = 0; s < kPool; ++s)
          sum += (x4[s] = std::exp(x4[s] - mx));
        for (int s = 0; s < kPool; ++s)
          acc += x4[s] / sum * o_ik[b + s][d];
        pn[d] = acc;
      }
      o_pooled.push_back(pn);
    }
    auto wd = MatVec(L.proj, xd);
    for (auto& v : wd)
      v /= std::sqrt(double(kId) * kIh);
    std::vector<double> sc(o_pooled.size());
    for (std::size_t pl = 0; pl < sc.size(); ++pl) {
      double acc = 0.0;
      for (int h = 0; h < kIh; ++h) {
        double dot = 0.0;
        for (int d = 0; d < kId; ++d)
          dot += iqd[h * kId + d] * o_pooled[pl][d];
        acc += std::max(dot, 0.0) * wd[h];
      }
      sc[pl] = acc;
    }
    std::vector<int> order(sc.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return sc[a] > sc[b]; });
    std::vector<int> vis;
    for (int i = 0; i < std::min<int>(sc.size(), kTop); ++i)
      for (int s = 0; s < kPool; ++s)
        vis.push_back(order[i] * kPool + s);
    for (int k = 0; k < (pos + 1) % kPool; ++k)
      vis.push_back(pos - k);
    std::vector<double> outd(kHeads * kQk);
    for (int h = 0; h < kHeads; ++h) {
      const auto qabs_d = MatVec(
          L.kb,
          std::vector<double>(qd.begin() + h * kQk, qd.begin() + (h + 1) * kQk),
          static_cast<std::uint64_t>(h) * kLat, kLat);
      std::vector<double> s_(vis.size());
      double mx = -1e300;
      for (std::size_t j = 0; j < vis.size(); ++j) {
        double dot = 0.0;
        for (int c = 0; c < kLat; ++c)
          dot += qabs_d[c] * o_lat[vis[j]][c];
        s_[j] = dot / std::sqrt(double(kQk));
        mx = std::max(mx, s_[j]);
      }
      double den = 0.0;
      for (auto& v : s_)
        den += (v = std::exp(v - mx));
      std::vector<double> ctx_d(kLat, 0.0);
      for (std::size_t j = 0; j < vis.size(); ++j)
        for (int c = 0; c < kLat; ++c)
          ctx_d[c] += s_[j] / den * o_lat[vis[j]][c];
      const auto oh =
          MatVec(L.vb, ctx_d, static_cast<std::uint64_t>(h) * kQk, kQk);
      std::copy(oh.begin(), oh.end(), outd.begin() + h * kQk);
    }
    const auto blockd = MatVec(L.out, outd);
    const double e = Rel(Host(block, kHidden), blockd);
    worst = std::max(worst, e);
    ok &= rc == 0 && e < kChainTolerance;
  }
  std::printf("layer %2d DSA chain %d tokens top=%d worst=%.2e %s\n", layer_idx,
              kTokens, kTop, worst, ok ? "OK" : "FAIL");
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
  bool ok = StageTests(*reader);
  ok &= ChainTest(*reader, 3);
  return ok ? 0 : 1;
}
