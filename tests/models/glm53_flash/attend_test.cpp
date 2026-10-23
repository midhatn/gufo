#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "src/models/glm53_flash/kernels/rocm/dsa.hpp"

namespace rocm = gufo::models::glm53_flash::rocm;

namespace {

constexpr int kHeads = rocm::kDsaHeads;
constexpr int kLat = rocm::kDsaLatent;
constexpr int kPool = rocm::kDsaPool;
constexpr int kTop = rocm::kDsaTopPools;

std::uint16_t ToHalfBits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t bits;
  std::memcpy(&bits, &h, sizeof bits);
  return bits;
}

float FromHalfBits(std::uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, sizeof h);
  return static_cast<float>(h);
}

}  // namespace

int main() {
  const int pos0 = 7168;
  const int tokens = 1024;
  const int rows = pos0 + tokens;
  std::mt19937 rng(11);
  std::normal_distribution<float> normal(0.0F, 1.0F);
  std::vector<std::uint16_t> lat(static_cast<std::size_t>(rows) * kLat);
  for (auto& x : lat) {
    x = ToHalfBits(0.3F * normal(rng));
  }
  std::vector<float> q(static_cast<std::size_t>(kHeads) * tokens * kLat);
  for (auto& x : q) {
    x = normal(rng);
  }
  std::vector<int> sel(static_cast<std::size_t>(tokens) * kTop);
  for (int t = 0; t < tokens; ++t) {
    const int npool = (pos0 + t + 1) / kPool;
    std::vector<int> pools(npool);
    std::iota(pools.begin(), pools.end(), 0);
    std::shuffle(pools.begin(), pools.end(), rng);
    const int n = std::min(npool, kTop);
    for (int i = 0; i < n; ++i) {
      sel[static_cast<std::size_t>(t) * kTop + i] = pools[i];
    }
  }
  std::uint16_t* d_lat = nullptr;
  float* d_q = nullptr;
  int* d_sel = nullptr;
  std::uint16_t* d_ctx = nullptr;
  hipMalloc(&d_lat, lat.size() * 2);
  hipMalloc(&d_q, q.size() * 4);
  hipMalloc(&d_sel, sel.size() * 4);
  hipMalloc(&d_ctx, static_cast<std::size_t>(kHeads) * tokens * kLat * 2);
  hipMemcpy(d_lat, lat.data(), lat.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_q, q.data(), q.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(d_sel, sel.data(), sel.size() * 4, hipMemcpyHostToDevice);

  const auto run = [&] {
    return rocm::DsaAttendBatch(d_q, d_lat, d_sel, pos0, tokens, kTop,
                                1.0F / 16.0F, d_ctx, nullptr);
  };
  run();
  hipDeviceSynchronize();
  std::vector<std::uint16_t> ctx(static_cast<std::size_t>(kHeads) * tokens *
                                 kLat);
  hipMemcpy(ctx.data(), d_ctx, ctx.size() * 2, hipMemcpyDeviceToHost);

  double worst = 0.0;
  for (const int t : {0, 5, 511, 1023}) {
    const int pos = pos0 + t;
    const int nsel = std::min((pos + 1) / kPool, kTop);
    std::vector<int> vis;
    for (int i = 0; i < nsel; ++i) {
      const int pool = sel[static_cast<std::size_t>(t) * kTop + i];
      for (int s = 0; s < kPool; ++s) {
        vis.push_back(pool * kPool + s);
      }
    }
    for (int i = 0; i < (pos + 1) % kPool; ++i) {
      vis.push_back(pos - i);
    }
    for (const int h : {0, 17, 40, 63}) {
      const float* qh = &q[(static_cast<std::size_t>(h) * tokens + t) * kLat];
      std::vector<double> score(vis.size());
      double best = -1e300;
      for (std::size_t j = 0; j < vis.size(); ++j) {
        double dot = 0.0;
        for (int c = 0; c < kLat; ++c) {
          dot += static_cast<double>(qh[c]) *
                 FromHalfBits(lat[static_cast<std::size_t>(vis[j]) * kLat + c]);
        }
        score[j] = dot / 16.0;
        best = std::max(best, score[j]);
      }
      double denom = 0.0;
      for (auto& s : score) {
        s = std::exp(s - best);
        denom += s;
      }
      for (int c = 0; c < kLat; c += 7) {
        double acc = 0.0;
        for (std::size_t j = 0; j < vis.size(); ++j) {
          acc += score[j] *
                 FromHalfBits(lat[static_cast<std::size_t>(vis[j]) * kLat + c]);
        }
        const double got = FromHalfBits(
            ctx[(static_cast<std::size_t>(h) * tokens + t) * kLat + c]);
        worst = std::max(worst, std::fabs(got - acc / denom));
      }
    }
  }
  hipDeviceSynchronize();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 5; ++i) {
    run();
  }
  hipDeviceSynchronize();
  const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count() /
                    5.0;
  std::printf("attend %d tokens: %.2f ms (%.1f us/token), worst %.3e\n", tokens,
              ms, 1000.0 * ms / tokens, worst);
  const bool ok = worst < 2e-3;
  std::printf(ok ? "OK\n" : "FAIL\n");
  return ok ? 0 : 1;
}
