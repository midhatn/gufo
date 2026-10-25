#ifndef GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_XOR_LANE_HPP_
#define GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_XOR_LANE_HPP_

#include <hip/hip_runtime.h>

namespace gufo::models::glm53_flash::rocm {

// Lane i receives lane (i ^ offset)'s value, as __shfl_xor does across a
// wave32: a DPP row_xmask (offsets below 16) or v_permlanex16 (16) instead of
// an LDS crossbar round trip. Every lane must be active.
__device__ __forceinline__ int XorLane(int v, int offset) {
#if defined(__gfx1151__)
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset, 32);
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

__device__ __forceinline__ double XorLane(double v, int offset) {
  const long long bits = __builtin_bit_cast(long long, v);
  const int lo = XorLane(static_cast<int>(bits), offset);
  const int hi = XorLane(static_cast<int>(bits >> 32), offset);
  return __builtin_bit_cast(double, (static_cast<long long>(hi) << 32) |
                                        static_cast<unsigned int>(lo));
}

}  // namespace gufo::models::glm53_flash::rocm

#endif  // GUFO_MODELS_GLM53_FLASH_KERNELS_ROCM_XOR_LANE_HPP_
