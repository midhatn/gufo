#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_NPU_DECODE_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_NPU_DECODE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// Decodes block-float C streams of the NPU GEMM (tokens x 1024 per slice) and
// adds the residual pass: out_t[token][(slice % per_tensor) * 1024 + c] =
// hi + lo for slice / per_tensor == t. hi / lo are device tables of one stream
// pointer per slice.
int NpuDecodeFused(const std::uint8_t* const* hi, const std::uint8_t* const* lo,
                   float* out0, float* out1, float* out2, int tokens,
                   int slices, int per_tensor, std::size_t pitch,
                   hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_NPU_DECODE_HPP_
