#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MATVEC_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MATVEC_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// ggml type codes accepted by Matvec.
inline constexpr int kTypeF32 = 0;
inline constexpr int kTypeQ8_0 = 8;
inline constexpr int kTypeQ6_K = 14;
inline constexpr int kTypeIQ2_XXS = 16;
inline constexpr int kTypeIQ2_S = 22;

// Device scratch the quantized-activation path needs for n columns of k.
std::size_t MatvecScratchBytes(int n, int k);

// out[col * m + row] = sum_k W[row, k] * x[col * k_dim + k] for n <= 8
// columns. W is a device tensor in its ggml block format with rows of k
// elements. Quantized types go through the vendored mmvq kernels; F32 uses a
// plain row-per-block kernel. scratch must hold MatvecScratchBytes(n, k).
// Returns 0 on success.
int Matvec(int ggml_type, const void* w, const float* x, float* out, int m,
           int n, int k, void* scratch, hipStream_t stream);

// Per-channel matvec: out[c * m + row] = sum_k W[c][row][k] * x[c * k + k'],
// for `channels` stacked weight slabs of m rows each (attn_k_b / attn_v_b
// heads). scratch must hold MatvecScratchBytes(channels, k). Returns 0 on
// success.
int MatvecBatched(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int channels, void* scratch,
                  hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_MATVEC_HPP_
