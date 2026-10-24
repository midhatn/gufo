#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/kda.hpp"

namespace rocm = gufo::models::glm53_flash::rocm;

namespace {

constexpr int kInner = rocm::kKdaInner;
constexpr int kHeads = rocm::kKdaHeads;
constexpr int kDim = rocm::kKdaHeadDim;

float* Upload(const std::vector<float>& v) {
  float* d = nullptr;
  hipMalloc(&d, v.size() * sizeof(float));
  hipMemcpy(d, v.data(), v.size() * sizeof(float), hipMemcpyHostToDevice);
  return d;
}

std::vector<float> Download(const float* d, std::size_t n) {
  std::vector<float> v(n);
  hipMemcpy(v.data(), d, n * sizeof(float), hipMemcpyDeviceToHost);
  return v;
}

bool Run(int tokens, bool timing, double decay_floor) {
  std::mt19937 rng(7);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  const auto fill = [&](std::size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) {
      x = normal(rng);
    }
    return v;
  };
  const std::size_t n = static_cast<std::size_t>(tokens) * kInner;
  const auto q = fill(n);
  const auto k = fill(n);
  const auto v = fill(n);
  std::vector<float> decay(n);
  for (auto& x : decay) {
    x = static_cast<float>(decay_floor + (1.0 - decay_floor) * unit(rng));
  }
  std::vector<float> beta(static_cast<std::size_t>(tokens) * kHeads);
  for (auto& x : beta) {
    x = unit(rng);
  }
  auto state0 = fill(static_cast<std::size_t>(kHeads) * kDim * kDim);
  for (auto& x : state0) {
    x *= 0.1F;
  }
  float *dq = Upload(q), *dk = Upload(k), *dv = Upload(v), *dd = Upload(decay),
        *db = Upload(beta);
  float *s1 = Upload(state0), *s2 = Upload(state0);
  float *o1 = nullptr, *o2 = nullptr;
  hipMalloc(&o1, n * sizeof(float));
  hipMalloc(&o2, n * sizeof(float));
  rocm::KdaRecur(dq, dk, dv, dd, db, s1, o1, nullptr, tokens);
  rocm::KdaRecurChunked(dq, dk, dv, dd, db, s2, o2, nullptr, tokens);
  hipDeviceSynchronize();
  const auto a = Download(o1, n);
  const auto b = Download(o2, n);
  const auto sa = Download(s1, state0.size());
  const auto sb = Download(s2, state0.size());
  double worst_o = 0.0;
  double scale_o = 0.0;
  double worst_s = 0.0;
  double scale_s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    worst_o = std::max(worst_o, static_cast<double>(std::fabs(a[i] - b[i])));
    scale_o = std::max(scale_o, static_cast<double>(std::fabs(a[i])));
  }
  for (std::size_t i = 0; i < sa.size(); ++i) {
    worst_s = std::max(worst_s, static_cast<double>(std::fabs(sa[i] - sb[i])));
    scale_s = std::max(scale_s, static_cast<double>(std::fabs(sa[i])));
  }
  std::printf(
      "tokens %d floor %.3f: o worst %.3e (scale %.3f)  state worst "
      "%.3e (scale %.3f)\n",
      tokens, decay_floor, worst_o, scale_o, worst_s, scale_s);
  if (timing) {
    const auto time = [&](auto&& fn) {
      fn();
      hipDeviceSynchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 3; ++i) {
        fn();
      }
      hipDeviceSynchronize();
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
                 .count() /
             3.0;
    };
    const double ms_old = time(
        [&] { rocm::KdaRecur(dq, dk, dv, dd, db, s1, o1, nullptr, tokens); });
    const double ms_new = time([&] {
      rocm::KdaRecurChunked(dq, dk, dv, dd, db, s2, o2, nullptr, tokens);
    });
    std::printf("  old %.2f ms  chunked %.2f ms\n", ms_old, ms_new);
  }
  return worst_o < 1e-3 * std::max(scale_o, 1.0) &&
         worst_s < 1e-3 * std::max(scale_s, 1.0);
}

}  // namespace

int main() {
  bool ok = Run(1024, true, 0.0067);
  ok = Run(1000, false, 0.0067) && ok;
  ok = Run(37, false, 0.5) && ok;
  std::printf(ok ? "OK\n" : "FAIL\n");
  return ok ? 0 : 1;
}
