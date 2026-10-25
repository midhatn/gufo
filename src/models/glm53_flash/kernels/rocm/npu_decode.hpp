#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_NPU_DECODE_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_NPU_DECODE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// Decodes block-float C streams of the NPU GEMM (tokens x 1024 per slice):
// out_t[token][(slice % per_tensor) * 1024 + c] for slice / per_tensor == t.
// `streams` is a device table of one stream pointer per slice.
int NpuDecodeFused(const std::uint8_t* const* streams, float* out0, float* out1,
                   float* out2, int tokens, int slices, int per_tensor,
                   std::size_t pitch, hipStream_t stream);

// The KDA short convolution (4 taps, SiLU) over q, k and v read straight from
// the NPU streams (24 slices: tensor * 8 + 1024-column slice, tokens == the
// stream row count), writing the convolved F32 rows and updating the
// [3 tensors][3 taps][8192] history.
int KdaConvStreams(const std::uint8_t* const* streams, const float* conv_q,
                   const float* conv_k, const float* conv_v, float* history,
                   float* out_q, float* out_k, float* out_v, int tokens,
                   hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_NPU_DECODE_HPP_
