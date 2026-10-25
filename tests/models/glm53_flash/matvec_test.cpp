#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::core::GgufTensorInfo;
namespace rocm = gufo::models::glm53_flash::rocm;

constexpr double kTolerance = 0.03;
constexpr int kSkip = 77;
constexpr std::uint64_t kMaxRows = 512;

int TypeCode(GgmlType t) {
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

void DequantRow(const GgufTensorInfo& t, const char* src, float* out,
                std::uint64_t k) {
  switch (t.type) {
    case GgmlType::kF32:
      std::copy_n(reinterpret_cast<const float*>(src), k, out);
      break;
    case GgmlType::kQ8_0:
      gufo::quant::DequantizeQ8_0(src, out, k);
      break;
    case GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out, k);
      break;
    case GgmlType::kIQ2_XXS:
      gufo::quant::DequantizeIQ2_XXS(src, out, k);
      break;
    case GgmlType::kIQ2_S:
      gufo::quant::DequantizeIQ2_S(src, out, k);
      break;
    default:
      std::abort();
  }
}

// Checks the first rows of one slab (expert) of a tensor, n activation columns.
bool Check(const GgufTensorInfo& t, std::uint64_t slab, int n) {
  const std::uint64_t k = t.dimensions[0];
  const std::uint64_t slab_rows = t.dimensions[1];
  const std::uint64_t rows = std::min(slab_rows, kMaxRows);
  const std::uint64_t row_bytes =
      t.size_bytes /
      (slab_rows * (t.dimensions.size() > 2 ? t.dimensions[2] : 1));
  const char* base =
      static_cast<const char*>(t.data) + slab * slab_rows * row_bytes;

  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> x(k * n);
  for (auto& v : x)
    v = dist(rng);

  std::vector<float> ref(rows * n);
  std::vector<float> w(k);
  for (std::uint64_t r = 0; r < rows; ++r) {
    DequantRow(t, base + r * row_bytes, w.data(), k);
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (std::uint64_t i = 0; i < k; ++i)
        acc += w[i] * x[c * k + i];
      ref[c * rows + r] = static_cast<float>(acc);
    }
  }

  void *dw = nullptr, *dx = nullptr, *dout = nullptr, *dscratch = nullptr;
  const std::size_t wbytes = rows * row_bytes;
  const std::size_t scratch = rocm::MatvecScratchBytes(n, static_cast<int>(k));
  if (hipMalloc(&dw, wbytes) || hipMalloc(&dx, x.size() * 4) ||
      hipMalloc(&dout, ref.size() * 4) || hipMalloc(&dscratch, scratch)) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return false;
  }
  hipMemcpy(dw, base, wbytes, hipMemcpyHostToDevice);
  hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
  hipMemset(dout, 0, ref.size() * 4);
  const int rc = rocm::Matvec(TypeCode(t.type), dw, static_cast<float*>(dx),
                              static_cast<float*>(dout), static_cast<int>(rows),
                              n, static_cast<int>(k), dscratch, nullptr);
  hipDeviceSynchronize();
  std::vector<float> got(ref.size());
  hipMemcpy(got.data(), dout, got.size() * 4, hipMemcpyDeviceToHost);
  hipFree(dw);
  hipFree(dx);
  hipFree(dout);
  hipFree(dscratch);

  double err2 = 0.0, ref2 = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double d = got[i] - ref[i];
    err2 += d * d;
    ref2 += static_cast<double>(ref[i]) * ref[i];
  }
  const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
  const bool ok = rc == 0 && rel < kTolerance;
  std::printf("%-34.*s type=%d k=%llu rows=%llu n=%d rc=%d rel=%.3e %s\n",
              static_cast<int>(t.name.size()), t.name.data(), TypeCode(t.type),
              static_cast<unsigned long long>(k),
              static_cast<unsigned long long>(rows), n, rc, rel,
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
  bool ok = true;
  const char* names[] = {"blk.0.attn_q.weight",   // Q6_K
                         "blk.0.ssm_f_b.weight",  // Q8_0, k = 128
                         "blk.0.ssm_f_a.weight",  // Q8_0, k = 4096
                         "blk.3.indexer.attn_q_b.weight",
                         "blk.3.ffn_gate_exps.weight",  // IQ2_XXS
                         "blk.3.ffn_down_exps.weight",  // IQ2_XXS or IQ2_S
                         "blk.3.ffn_gate_inp.weight"};  // F32
  for (const char* name : names) {
    const auto* t = reader->FindTensor(name);
    if (t == nullptr) {
      std::printf("missing %s\n", name);
      ok = false;
      continue;
    }
    for (const int n : {1, 4}) {
      ok &= Check(*t, t->dimensions.size() > 2 ? 5 : 0, n);
    }
  }
  for (std::uint32_t layer = 3; layer < 45; ++layer) {
    const auto* t = reader->FindTensor("blk." + std::to_string(layer) +
                                       ".ffn_down_exps.weight");
    if (t != nullptr && t->type == GgmlType::kIQ2_S) {
      ok &= Check(*t, 7, 1);
      break;
    }
  }
  return ok ? 0 : 1;
}
