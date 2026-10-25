#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace qfn = gufo::models::qwen38_flash_next::rocm;

int main() {
  const int tokens = 4096;
  struct Shape {
    const char* name;
    std::size_t m, k;
  } shapes[] = {{"qkv slice 8192x4096", 8192, 4096},
                {"kda out 4096x8192", 4096, 8192},
                {"shexp down 4096x2048", 4096, 2048},
                {"shexp gate 2048x4096", 2048, 4096},
                {"q_b 16384x1536", 16384, 1536},
                {"attn out 4096x16384", 4096, 16384},
                {"dense ffn gate 12288x4096", 12288, 4096},
                {"dense ffn down 4096x12288", 4096, 12288}};
  for (const auto& s : shapes) {
    const std::size_t wbytes = s.m * (s.k / 32) * 34;
    void* w = nullptr;
    hipMalloc(&w, wbytes);
    hipMemset(w, 0x11, wbytes);
    float* x = nullptr;
    hipMalloc(&x, std::size_t{tokens} * s.k * 4);
    hipMemset(x, 0, std::size_t{tokens} * s.k * 4);
    __half* xh = nullptr;
    hipMalloc(&xh, std::size_t{tokens} * s.k * 2);
    hipMemset(xh, 0, std::size_t{tokens} * s.k * 2);
    float* out = nullptr;
    hipMalloc(&out, std::size_t{tokens} * s.m * 4);
    void* q = nullptr;
    hipMalloc(&q, qfn::Q8TiledBytes(tokens, s.k) + 4096);
    qfn::QuantizeQ8Tiled(x, q, tokens, s.k, nullptr);
    hipDeviceSynchronize();
    const auto time = [&](auto&& fn) {
      fn();
      hipDeviceSynchronize();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 5; ++i) {
        fn();
      }
      hipDeviceSynchronize();
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
                 .count() /
             5.0;
    };
    const double flops = 2.0 * tokens * s.m * s.k;
    bool ok_w8 = true;
    bool ok_f16 = true;
    const double t_w8 = time([&] {
      ok_w8 = qfn::W8A8Gemm(w, q, out, tokens, s.m, s.k, nullptr) && ok_w8;
    });
    const double t_f16 = time([&] {
      ok_f16 =
          qfn::DenseF16Gemm(w, xh, out, tokens, s.m, s.k, nullptr) && ok_f16;
    });
    std::printf("%-28s w8a8 %s %.2f ms (%.1f T)   f16 %s %.2f ms (%.1f T)\n",
                s.name, ok_w8 ? "ok" : "NO", t_w8, flops / t_w8 / 1e9,
                ok_f16 ? "ok" : "NO", t_f16, flops / t_f16 / 1e9);
    hipFree(w);
    hipFree(x);
    hipFree(xh);
    hipFree(out);
    hipFree(q);
  }
  return 0;
}
