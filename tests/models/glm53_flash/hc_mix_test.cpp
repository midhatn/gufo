#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"
#include "src/models/glm53_flash/kernels/rocm/router.hpp"

namespace rocm = gufo::models::glm53_flash::rocm;

namespace {

template<typename T>
T* Upload(const std::vector<T>& v) {
  T* d = nullptr;
  hipMalloc(&d, v.size() * sizeof(T));
  hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

std::uint16_t Bits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t b;
  std::memcpy(&b, &h, sizeof b);
  return b;
}

}  // namespace

int main() {
  constexpr int kTokens = 100;
  constexpr int kFlat = rocm::kHcStreams * rocm::kHcHidden;
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(kTokens) * kFlat);
  for (auto& v : x) {
    v = Bits(normal(rng) * 3.0F);
  }
  std::vector<float> fn(static_cast<std::size_t>(rocm::kHcMixes) * kFlat);
  for (auto& v : fn) {
    v = normal(rng) * 0.002F;
  }
  std::vector<float> scale = {0.5F, 0.5F, 0.5F};
  std::vector<float> base(rocm::kHcMixes, 0.1F);
  std::vector<float> norm_w(rocm::kHcHidden, 1.0F);
  auto* dx = Upload(x);
  auto* dfn = Upload(fn);
  auto* dscale = Upload(scale);
  auto* dbase = Upload(base);
  auto* dnorm = Upload(norm_w);
  std::uint16_t *hi = nullptr, *lo = nullptr;
  hipMalloc(&hi, fn.size() * 2);
  hipMalloc(&lo, fn.size() * 2);
  rocm::SplitF16(dfn, hi, lo, static_cast<long long>(fn.size()), nullptr,
                 rocm::kHcMixFnScale);
  float *y1, *y2, *p1, *p2, *c1, *c2, *s1, *s2;
  for (float** p : {&y1, &y2}) {
    hipMalloc(p, std::size_t{kTokens} * rocm::kHcHidden * 4);
  }
  for (float** p : {&p1, &p2}) {
    hipMalloc(p, std::size_t{kTokens} * rocm::kHcStreams * 4);
  }
  for (float** p : {&c1, &c2}) {
    hipMalloc(p, std::size_t{kTokens} * 16 * 4);
  }
  for (float** p : {&s1, &s2}) {
    hipMalloc(p, std::size_t{kTokens} * rocm::kHcPreScratch * 4);
  }
  rocm::HcPre(dx, dfn, dscale, dbase, dnorm, 1e-6F, y1, p1, c1, s1, nullptr,
              kTokens, nullptr);
  rocm::HcPre(dx, dfn, dscale, dbase, dnorm, 1e-6F, y2, p2, c2, s2, nullptr,
              kTokens, nullptr, hi, lo);
  hipDeviceSynchronize();
  const auto get = [](const float* d, std::size_t n) {
    std::vector<float> v(n);
    hipMemcpy(v.data(), d, n * 4, hipMemcpyDeviceToHost);
    return v;
  };
  const auto worst = [](const std::vector<float>& a,
                        const std::vector<float>& b) {
    double w = 0.0, s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      w = std::max(w, static_cast<double>(std::fabs(a[i] - b[i])));
      s = std::max(s, static_cast<double>(std::fabs(a[i])));
    }
    return std::pair<double, double>(w, s);
  };
  const auto sa = get(s1, std::size_t{kTokens} * rocm::kHcPreScratch);
  const auto sb = get(s2, std::size_t{kTokens} * rocm::kHcPreScratch);
  const auto [ws, ss] = worst(sa, sb);
  const auto [wy, sy] = worst(get(y1, std::size_t{kTokens} * rocm::kHcHidden),
                              get(y2, std::size_t{kTokens} * rocm::kHcHidden));
  const auto [wc, sc] = worst(get(c1, std::size_t{kTokens} * 16),
                              get(c2, std::size_t{kTokens} * 16));
  std::printf(
      "scratch worst %.3e (scale %.3f)  y worst %.3e (scale %.3f)  comb "
      "worst %.3e\n",
      ws, ss, wy, sy, wc);
  const bool ok = ws < 1e-3 * ss && wy < 1e-3 * sy && wc < 1e-4;
  std::printf(ok ? "OK\n" : "FAIL\n");
  return ok ? 0 : 1;
}
