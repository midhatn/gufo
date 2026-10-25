#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

namespace rocm = gufo::models::glm53_flash::rocm;

constexpr int kHc = rocm::kHcStreams;
constexpr int kH = rocm::kHcHidden;
constexpr double kTolerance = 2e-5;
constexpr int kSkip = 77;
constexpr float kEps = 1e-6F;
constexpr float kRmsEps = 1e-5F;
constexpr int kSinkhornRounds = 19;

double Sigmoid(double v) {
  return 1.0 / (1.0 + std::exp(-v));
}

struct Oracle {
  std::vector<double> y, post, comb;
};

// Double-precision restatement of the hc_pre contract in glm53_flash/AGENTS.md.
Oracle HcPreOracle(const std::vector<float>& x, const float* fn,
                   const float* scale, const float* base, const float* norm_w) {
  const int flat = kHc * kH;
  double ss = 0.0;
  for (const float v : x)
    ss += static_cast<double>(v) * v;
  const double inv = 1.0 / std::sqrt(ss / flat + kRmsEps);
  std::vector<double> mix(24, 0.0);
  for (int j = 0; j < 24; ++j) {
    for (int i = 0; i < flat; ++i)
      mix[j] += fn[j * flat + i] * x[i] * inv;
  }
  Oracle o;
  std::vector<double> pre(kHc);
  o.post.resize(kHc);
  for (int s = 0; s < kHc; ++s) {
    pre[s] = Sigmoid(mix[s] * scale[0] + base[s]) + kEps;
    o.post[s] = Sigmoid(mix[4 + s] * scale[1] + base[4 + s]) * 2.0;
  }
  // comb[dst][src] at mixes[8 + dst + 4 * src].
  double c[kHc][kHc];
  for (int d = 0; d < kHc; ++d)
    for (int s = 0; s < kHc; ++s)
      c[d][s] = mix[8 + d + 4 * s] * scale[2] + base[8 + d + 4 * s];
  for (int s = 0; s < kHc; ++s) {
    double mx = -1e300, sum = 0.0;
    for (int d = 0; d < kHc; ++d)
      mx = std::max(mx, c[d][s]);
    for (int d = 0; d < kHc; ++d)
      sum += (c[d][s] = std::exp(c[d][s] - mx));
    for (int d = 0; d < kHc; ++d)
      c[d][s] /= sum + kEps;
  }
  const auto norm_src = [&] {
    for (int d = 0; d < kHc; ++d) {
      double sum = 0.0;
      for (int s = 0; s < kHc; ++s)
        sum += c[d][s];
      for (int s = 0; s < kHc; ++s)
        c[d][s] /= sum + kEps;
    }
  };
  const auto norm_dst = [&] {
    for (int s = 0; s < kHc; ++s) {
      double sum = 0.0;
      for (int d = 0; d < kHc; ++d)
        sum += c[d][s];
      for (int d = 0; d < kHc; ++d)
        c[d][s] /= sum + kEps;
    }
  };
  norm_src();
  for (int i = 0; i < kSinkhornRounds; ++i) {
    norm_dst();
    norm_src();
  }
  o.comb.resize(kHc * kHc);
  for (int d = 0; d < kHc; ++d)
    for (int s = 0; s < kHc; ++s)
      o.comb[d + kHc * s] = c[d][s];
  std::vector<double> z(kH);
  double zz = 0.0;
  for (int i = 0; i < kH; ++i) {
    for (int s = 0; s < kHc; ++s)
      z[i] += x[s * kH + i] * pre[s];
    zz += z[i] * z[i];
  }
  const double zi = 1.0 / std::sqrt(zz / kH + kRmsEps);
  o.y.resize(kH);
  for (int i = 0; i < kH; ++i)
    o.y[i] = z[i] * zi * norm_w[i];
  return o;
}

double Rel(const std::vector<float>& got, const std::vector<double>& ref) {
  double e = 0.0, r = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    e += (got[i] - ref[i]) * (got[i] - ref[i]);
    r += ref[i] * ref[i];
  }
  return std::sqrt(e / std::max(r, 1e-30));
}

template<typename T>
T* ToDevice(const std::vector<T>& v) {
  T* d = nullptr;
  hipMalloc(&d, v.size() * sizeof(T));
  hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

std::vector<float> FromDevice(const float* d, std::size_t n) {
  std::vector<float> v(n);
  hipMemcpy(v.data(), d, n * sizeof(float), hipMemcpyDeviceToHost);
  return v;
}

bool CheckLayer(const gufo::core::GgufReader& reader, int layer,
                const char* kind, std::mt19937& rng) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto* fn = reader.FindTensor(p + "hc_" + kind + "_fn.weight");
  const auto* sc = reader.FindTensor(p + "hc_" + kind + "_scale.weight");
  const auto* bs = reader.FindTensor(p + "hc_" + kind + "_base.weight");
  const auto* nw = reader.FindTensor(
      p + (std::string(kind) == "attn" ? "attn" : "ffn") + "_norm.weight");
  if (!fn || !sc || !bs || !nw) {
    std::printf("missing hc tensors for layer %d\n", layer);
    return false;
  }
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> x(kHc * kH);
  for (int s = 0; s < kHc; ++s)
    for (int i = 0; i < kH; ++i)
      x[s * kH + i] = dist(rng) * (1.0F + s);
  const float* fnp = static_cast<const float*>(fn->data);
  const Oracle o = HcPreOracle(x, fnp, static_cast<const float*>(sc->data),
                               static_cast<const float*>(bs->data),
                               static_cast<const float*>(nw->data));

  float* dx = ToDevice(x);
  float* dfn = ToDevice(std::vector<float>(fnp, fnp + 24 * kHc * kH));
  float* dsc =
      ToDevice(std::vector<float>(static_cast<const float*>(sc->data),
                                  static_cast<const float*>(sc->data) + 3));
  float* dbs =
      ToDevice(std::vector<float>(static_cast<const float*>(bs->data),
                                  static_cast<const float*>(bs->data) + 24));
  float* dnw =
      ToDevice(std::vector<float>(static_cast<const float*>(nw->data),
                                  static_cast<const float*>(nw->data) + kH));
  float *dy, *dpost, *dcomb;
  hipMalloc(&dy, kH * 4);
  hipMalloc(&dpost, kHc * 4);
  hipMalloc(&dcomb, kHc * kHc * 4);
  float* dscratch;
  hipMalloc(&dscratch, rocm::kHcPreScratch * 4);
  const int rc = rocm::HcPre(dx, dfn, dsc, dbs, dnw, kEps, dy, dpost, dcomb,
                             dscratch, nullptr);
  hipDeviceSynchronize();
  const double ey = Rel(FromDevice(dy, kH), o.y);
  const double ep = Rel(FromDevice(dpost, kHc), o.post);
  const double ec = Rel(FromDevice(dcomb, kHc * kHc), o.comb);

  // Post-mix: random block output.
  std::vector<float> block(kH);
  for (auto& v : block)
    v = dist(rng);
  float* dblock = ToDevice(block);
  float* dnext;
  hipMalloc(&dnext, kHc * kH * 4);
  const int rc2 = rocm::HcPost(dx, dblock, dpost, dcomb, dnext, nullptr);
  hipDeviceSynchronize();
  std::vector<double> ref_next(kHc * kH);
  for (int d = 0; d < kHc; ++d)
    for (int i = 0; i < kH; ++i) {
      double acc = block[i] * o.post[d];
      for (int s = 0; s < kHc; ++s)
        acc += x[s * kH + i] * o.comb[d + kHc * s];
      ref_next[d * kH + i] = acc;
    }
  const double en = Rel(FromDevice(dnext, kHc * kH), ref_next);
  const bool ok = rc == 0 && rc2 == 0 && ey < kTolerance && ep < kTolerance &&
                  ec < kTolerance && en < kTolerance;
  std::printf("layer %2d hc_%-4s y=%.2e post=%.2e comb=%.2e next=%.2e %s\n",
              layer, kind, ey, ep, ec, en, ok ? "OK" : "FAIL");
  for (void* p : {static_cast<void*>(dx), static_cast<void*>(dfn),
                  static_cast<void*>(dsc), static_cast<void*>(dbs),
                  static_cast<void*>(dnw), static_cast<void*>(dy),
                  static_cast<void*>(dpost), static_cast<void*>(dcomb),
                  static_cast<void*>(dblock), static_cast<void*>(dnext)})
    hipFree(p);
  return ok;
}

bool CheckHead(const gufo::core::GgufReader& reader, std::mt19937& rng) {
  const auto* w = reader.FindTensor("output_norm.weight");
  if (!w)
    return false;
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> x(kHc * kH);
  for (auto& v : x)
    v = dist(rng);
  std::vector<double> ref(kH);
  double ss = 0.0;
  for (int i = 0; i < kH; ++i) {
    double m = 0.0;
    for (int s = 0; s < kHc; ++s)
      m += x[s * kH + i];
    ref[i] = m / kHc;
    ss += ref[i] * ref[i];
  }
  const double inv = 1.0 / std::sqrt(ss / kH + kRmsEps);
  for (int i = 0; i < kH; ++i)
    ref[i] *= inv * static_cast<const float*>(w->data)[i];
  float* dx = ToDevice(x);
  float* dw =
      ToDevice(std::vector<float>(static_cast<const float*>(w->data),
                                  static_cast<const float*>(w->data) + kH));
  float* dout;
  hipMalloc(&dout, kH * 4);
  const int rc = rocm::HeadNorm(dx, dw, dout, nullptr);
  hipDeviceSynchronize();
  const double e = Rel(FromDevice(dout, kH), ref);
  const bool ok = rc == 0 && e < kTolerance;
  std::printf("head_norm rel=%.2e %s\n", e, ok ? "OK" : "FAIL");
  hipFree(dx);
  hipFree(dw);
  hipFree(dout);
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
  std::mt19937 rng(11);
  bool ok = true;
  for (const int layer : {0, 3, 44}) {
    ok &= CheckLayer(*reader, layer, "attn", rng);
    ok &= CheckLayer(*reader, layer, "ffn", rng);
  }
  ok &= CheckHead(*reader, rng);
  return ok ? 0 : 1;
}
