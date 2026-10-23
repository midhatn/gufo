// Times the grouped routed-expert GEMM on real layer weights:
//   glm53_experts_bench MODEL [layer]
// Reports per-projection time, TFLOP/s and weight-stream GB/s per token count.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

extern "C" void ds4_mmq_set_routed_tile_cols(int cols);
extern "C" void ds4_mmq_set_routed_max_expert_rows(int rows);
extern "C" int ds4_mmq_routed_tile_cols_for_counts(const unsigned int* counts,
                                                   int n_experts);

namespace {

namespace rocm = gufo::models::glm53_flash::rocm;
constexpr int kUsed = 8;
constexpr int kIters = 8;

int Code(gufo::core::GgmlType t) {
  switch (t) {
    case gufo::core::GgmlType::kIQ2_XXS:
      return rocm::kTypeIQ2_XXS;
    case gufo::core::GgmlType::kIQ2_S:
      return rocm::kTypeIQ2_S;
    default:
      return -1;
  }
}

void Bench(const gufo::core::GgufTensorInfo& t, int tokens, int used,
           const char* what) {
  const int k = static_cast<int>(t.dimensions[0]);
  const int m = static_cast<int>(t.dimensions[1]);
  const int experts = static_cast<int>(t.dimensions[2]);
  std::mt19937 rng(3);
  std::vector<float> x(static_cast<std::size_t>(tokens) * k);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  for (auto& v : x)
    v = nd(rng);
  std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * used);
  std::uniform_int_distribution<int> pick(0, experts - 1);
  for (int i = 0; i < tokens; ++i) {
    for (int s = 0; s < used; ++s) {
      int e;
      bool dup;
      do {
        e = pick(rng);
        dup = false;
        for (int q = 0; q < s; ++q)
          dup |= ids[i * used + q] == e;
      } while (dup);
      ids[i * used + s] = e;
    }
  }
  void* dw;
  hipMalloc(&dw, t.size_bytes + 4096);
  hipMemcpy(dw, t.data, t.size_bytes, hipMemcpyHostToDevice);
  float *dx, *dout;
  int* dids;
  hipMalloc(&dx, x.size() * 4);
  hipMalloc(&dout, static_cast<std::size_t>(m) * tokens * used * 4);
  hipMalloc(&dids, ids.size() * 4);
  hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice);
  const int code = Code(t.type);
  {
    std::vector<unsigned int> counts(experts, 0);
    for (const auto e : ids)
      ++counts[e];
    const int max_rows =
        static_cast<int>(*std::max_element(counts.begin(), counts.end()));
    const int picked =
        ds4_mmq_routed_tile_cols_for_counts(counts.data(), experts);
    int cols = picked;
    if (const char* v = std::getenv("GUFO_BENCH_COLS"))
      cols = std::atoi(v);
    std::printf("  [max bucket %d, model-picked tile %d, using %d]\n", max_rows,
                picked, cols);
    ds4_mmq_set_routed_max_expert_rows(max_rows);
    ds4_mmq_set_routed_tile_cols(cols);
  }
  rocm::ExpertsGemm(code, dw, dx, dout, m, k, tokens, experts, used, dids,
                    nullptr);
  hipDeviceSynchronize();
  hipEvent_t a, b;
  hipEventCreate(&a);
  hipEventCreate(&b);
  hipEventRecord(a);
  for (int i = 0; i < kIters; ++i) {
    rocm::ExpertsGemm(code, dw, dx, dout, m, k, tokens, experts, used, dids,
                      nullptr);
  }
  hipEventRecord(b);
  hipEventSynchronize(b);
  float ms = 0;
  hipEventElapsedTime(&ms, a, b);
  ms /= kIters;
  const double flops = 2.0 * m * k * tokens * used;
  std::printf(
      "%-6s T=%-5d rows=%-6d M=%-5d K=%-5d %7.3f ms  %6.2f TFLOP/s  weights "
      "%6.1f GB/s\n",
      what, tokens, tokens * used, m, k, ms, flops / ms / 1e9,
      static_cast<double>(t.size_bytes) / ms / 1e6);
  hipFree(dw);
  hipFree(dx);
  hipFree(dout);
  hipFree(dids);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2)
    return 77;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  if (!reader)
    return 77;
  const std::string layer = argc > 2 ? argv[2] : "3";
  const auto* gate =
      reader->FindTensor("blk." + layer + ".ffn_gate_exps.weight");
  const auto* down =
      reader->FindTensor("blk." + layer + ".ffn_down_exps.weight");
  if (!gate || !down)
    return 1;
  std::vector<int> sizes = {64, 128, 256, 512, 1024};
  if (const char* v = std::getenv("GUFO_BENCH_T"))
    sizes = {std::atoi(v)};
  for (const int tokens : sizes) {
    Bench(*gate, tokens, kUsed, "gate");
    Bench(*down, tokens * kUsed, 1, "down");
  }
  return 0;
}
