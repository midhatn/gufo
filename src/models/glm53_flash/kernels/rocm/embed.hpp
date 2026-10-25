#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EMBED_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EMBED_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

// Embedding rows of `tokens` (device ids) written to the four hyper-connection
// streams of out[t][4][4096], as F16 bits or F32. `type` is a core::GgmlType
// value of a Q-quantized or F32 table whose rows are `row_bytes` long.
int EmbedTokens(const void* table, std::uint32_t type, std::size_t row_bytes,
                const std::int32_t* tokens, void* out, bool half, int count,
                hipStream_t stream);

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_EMBED_HPP_
