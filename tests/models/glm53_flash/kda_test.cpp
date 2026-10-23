#include "src/models/glm53_flash/kernels/rocm/kda.hpp"

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

constexpr int kHeads = rocm::kKdaHeads;
constexpr int kD = rocm::kKdaHeadDim;
constexpr int kInner = rocm::kKdaInner;
constexpr int kSteps = 4;
constexpr double kTolerance = 1e-4;
constexpr int kSkip = 77;

double Sigmoid(double v) {
  return 1.0 / (1.0 + std::exp(-v));
}

struct Weights {
  const float *cq, *ck, *cv, *dt, *a, *norm;
};

struct State {
  std::vector<double> hist;  // [3][3][inner]
  std::vector<double> s;     // [heads][d][d]
};

// Double-precision statement of the KDA decode contract (AGENTS.md).
std::vector<double> OracleStep(const Weights& w, State& st,
                               const std::vector<float>& raw,
                               const std::vector<float>& f,
                               const std::vector<float>& braw,
                               const std::vector<float>& g) {
  std::vector<double> qkv(3 * kInner);
  const float* cws[3] = {w.cq, w.ck, w.cv};
  for (int s = 0; s < 3; ++s) {
    for (int ch = 0; ch < kInner; ++ch) {
      const float* tw = cws[s] + ch * 4;
      double acc = tw[3] * static_cast<double>(raw[s * kInner + ch]);
      for (int tap = 0; tap < 3; ++tap)
        acc += tw[tap] * st.hist[(s * 3 + tap) * kInner + ch];
      qkv[s * kInner + ch] = acc * Sigmoid(acc);
      st.hist[(s * 3 + 0) * kInner + ch] = st.hist[(s * 3 + 1) * kInner + ch];
      st.hist[(s * 3 + 1) * kInner + ch] = st.hist[(s * 3 + 2) * kInner + ch];
      st.hist[(s * 3 + 2) * kInner + ch] = raw[s * kInner + ch];
    }
  }
  std::vector<double> out(kInner);
  for (int h = 0; h < kHeads; ++h) {
    double q[kD], k[kD];
    double qq = 0.0, kk = 0.0;
    for (int i = 0; i < kD; ++i) {
      q[i] = qkv[h * kD + i];
      k[i] = qkv[kInner + h * kD + i];
      qq += q[i] * q[i];
      kk += k[i] * k[i];
    }
    for (int i = 0; i < kD; ++i) {
      q[i] /= std::sqrt(qq + 1e-6);
      k[i] /= std::sqrt(kk + 1e-6);
    }
    double decay[kD];
    for (int i = 0; i < kD; ++i) {
      const double fi = static_cast<double>(f[h * kD + i]) + w.dt[h * kD + i];
      decay[i] = std::exp(-5.0 * Sigmoid(-fi * w.a[h]));
    }
    const double beta = Sigmoid(braw[h]);
    double o[kD];
    for (int j = 0; j < kD; ++j) {
      double* row = &st.s[(static_cast<std::size_t>(h) * kD + j) * kD];
      double u = 0.0;
      for (int i = 0; i < kD; ++i) {
        row[i] *= decay[i];
        u += row[i] * k[i];
      }
      const double delta = (qkv[2 * kInner + h * kD + j] - u) * beta;
      double acc = 0.0;
      for (int i = 0; i < kD; ++i) {
        row[i] += delta * k[i];
        acc += row[i] * q[i];
      }
      o[j] = acc / std::sqrt(static_cast<double>(kD));
    }
    double ss = 0.0;
    for (int j = 0; j < kD; ++j)
      ss += o[j] * o[j];
    const double inv = 1.0 / std::sqrt(ss / kD + 1e-5);
    for (int j = 0; j < kD; ++j)
      out[h * kD + j] = o[j] * inv * w.norm[j] * Sigmoid(g[h * kD + j]);
  }
  return out;
}

float* Dev(const std::vector<float>& v) {
  float* d = nullptr;
  hipMalloc(&d, v.size() * 4);
  hipMemcpy(d, v.data(), v.size() * 4, hipMemcpyHostToDevice);
  return d;
}

float* Dev(const float* p, std::size_t n) {
  return Dev(std::vector<float>(p, p + n));
}

std::vector<float> Host(const float* d, std::size_t n) {
  std::vector<float> v(n);
  hipMemcpy(v.data(), d, n * 4, hipMemcpyDeviceToHost);
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

bool CheckLayer(const gufo::core::GgufReader& reader, int layer) {
  const std::string p = "blk." + std::to_string(layer) + ".";
  const auto get = [&](const char* n) -> const float* {
    const auto* t = reader.FindTensor(p + n);
    return t ? static_cast<const float*>(t->data) : nullptr;
  };
  Weights w{get("ssm_conv1d_q.weight"),
            get("ssm_conv1d_k.weight"),
            get("ssm_conv1d_v.weight"),
            get("ssm_dt.bias"),
            get("ssm_a"),
            get("ssm_norm.weight")};
  if (!w.cq || !w.ck || !w.cv || !w.dt || !w.a || !w.norm) {
    std::printf("missing KDA tensors in layer %d\n", layer);
    return false;
  }
  std::mt19937 rng(100 + layer);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  const auto rand_vec = [&](std::size_t n, float scale) {
    std::vector<float> v(n);
    for (auto& x : v)
      x = dist(rng) * scale;
    return v;
  };

  State st;
  st.hist.resize(9 * kInner);
  st.s.resize(static_cast<std::size_t>(kHeads) * kD * kD);
  const auto hist0 = rand_vec(9 * kInner, 1.0F);
  const auto s0 = rand_vec(st.s.size(), 0.1F);
  std::copy(hist0.begin(), hist0.end(), st.hist.begin());
  std::copy(s0.begin(), s0.end(), st.s.begin());

  float* dhist = Dev(hist0);
  float* dstate = Dev(s0);
  float* dcq = Dev(w.cq, 4 * kInner);
  float* dck = Dev(w.ck, 4 * kInner);
  float* dcv = Dev(w.cv, 4 * kInner);
  float* ddt = Dev(w.dt, kInner);
  float* da = Dev(w.a, kHeads);
  float* dnorm = Dev(w.norm, kD);
  float *dqkv, *ddecay, *dbeta, *do_, *df, *dbraw, *dg;
  hipMalloc(&dqkv, 3 * kInner * 4);
  hipMalloc(&ddecay, kInner * 4);
  hipMalloc(&dbeta, kHeads * 4);
  hipMalloc(&do_, kInner * 4);
  hipMalloc(&df, kInner * 4);
  hipMalloc(&dbraw, kHeads * 4);
  hipMalloc(&dg, kInner * 4);

  bool ok = true;
  double worst = 0.0;
  for (int step = 0; step < kSteps; ++step) {
    const auto raw = rand_vec(3 * kInner, 1.0F);
    const auto f = rand_vec(kInner, 2.0F);
    const auto braw = rand_vec(kHeads, 1.0F);
    const auto g = rand_vec(kInner, 1.0F);
    const auto ref = OracleStep(w, st, raw, f, braw, g);

    hipMemcpy(dqkv, raw.data(), raw.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(df, f.data(), f.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dbraw, braw.data(), braw.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice);
    int rc = rocm::KdaConv(dqkv, dqkv + kInner, dqkv + 2 * kInner, dhist, dcq,
                           dck, dcv, nullptr);
    rc |= rocm::KdaGates(df, ddt, da, dbraw, ddecay, dbeta, nullptr);
    rc |= rocm::KdaRecur(dqkv, dqkv + kInner, dqkv + 2 * kInner, ddecay, dbeta,
                         dstate, do_, nullptr);
    rc |= rocm::KdaOut(do_, dnorm, dg, nullptr);
    hipDeviceSynchronize();
    const double e = Rel(Host(do_, kInner), ref);
    worst = std::max(worst, e);
    ok &= rc == 0 && e < kTolerance;
  }
  const double es = Rel(Host(dstate, st.s.size()), st.s);
  const double eh = Rel(Host(dhist, st.hist.size()), st.hist);
  ok &= es < kTolerance && eh < 1e-7;
  std::printf("layer %2d KDA %d steps out=%.2e state=%.2e hist=%.2e %s\n",
              layer, kSteps, worst, es, eh, ok ? "OK" : "FAIL");
  for (float* d : {dhist, dstate, dcq, dck, dcv, ddt, da, dnorm, dqkv, ddecay,
                   dbeta, do_, df, dbraw, dg})
    hipFree(d);
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
  bool ok = true;
  for (const int layer : {0, 1, 2, 4, 40})
    ok &= CheckLayer(*reader, layer);
  return ok ? 0 : 1;
}
