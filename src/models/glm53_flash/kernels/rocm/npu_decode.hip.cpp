#include "src/models/glm53_flash/kernels/rocm/npu_decode.hpp"

namespace gufo::models::glm53_flash::rocm {

namespace {

// C tile geometry of the NPU kernel: 512 x 128 tiles of 2x2 super blocks of
// 8x8 value blocks, each block 9 bytes (shared exponent, eight int8).
constexpr int kTileRows = 512;
constexpr int kTileCols = 128;
constexpr int kGroupsPerTile = kTileRows * kTileCols / 8;
constexpr int kBlockGroups = 256;
constexpr int kStageBytes = kBlockGroups * 9;
constexpr int kStageVec = kStageBytes / 16;
constexpr int kSliceCols = 1024;

__global__ __launch_bounds__(kBlockGroups) void NpuDecodeFusedKernel(
    const std::uint8_t* const* __restrict__ hi_streams,
    const std::uint8_t* const* __restrict__ lo_streams, float* out0,
    float* out1, float* out2, int per_tensor, std::size_t pitch,
    bool residual) {
  const int slice = blockIdx.y;
  const std::size_t g0 = std::size_t{blockIdx.x} * kBlockGroups;
  __shared__ __attribute__((aligned(16))) std::uint8_t hi[kStageBytes];
  __shared__ __attribute__((aligned(16))) std::uint8_t lo[kStageBytes];
  const int t = threadIdx.x;
  if (t < kStageVec) {
    reinterpret_cast<uint4*>(hi)[t] =
        reinterpret_cast<const uint4*>(hi_streams[slice] + g0 * 9)[t];
    if (residual) {
      reinterpret_cast<uint4*>(lo)[t] =
          reinterpret_cast<const uint4*>(lo_streams[slice] + g0 * 9)[t];
    }
  }
  __syncthreads();
  const std::size_t tile = g0 / kGroupsPerTile;
  const int sr = static_cast<int>((g0 % kGroupsPerTile) / kBlockGroups);
  constexpr int kTilesN = kSliceCols / kTileCols;
  const std::size_t tile_row = tile / kTilesN;
  const int tile_col = static_cast<int>(tile % kTilesN);
  const int row_in = t >> 4;
  const int cc = t & 15;
  const int br = row_in >> 3;
  const int rib = row_in & 7;
  const int bc = cc & 1;
  const int sc = cc >> 1;
  const int gi = rib | (bc << 3) | (br << 4) | (sc << 5);
  const std::uint8_t* ph = hi + gi * 9;
  const std::uint8_t* pl = lo + gi * 9;
  const float mult_hi = ldexpf(1.0F, static_cast<int>(ph[0]) - 127) / 64.0F;
  const float mult_lo =
      residual ? ldexpf(1.0F, static_cast<int>(pl[0]) - 127) / 64.0F : 0.0F;
  const std::size_t row =
      tile_row * kTileRows + static_cast<std::size_t>((sr * 2 + br) * 8 + rib);
  const int col = tile_col * kTileCols + cc * 8;
  const int tensor = slice / per_tensor;
  float* out = tensor == 0 ? out0 : (tensor == 1 ? out1 : out2);
  float* dst = out + row * pitch + (slice % per_tensor) * kSliceCols + col;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = static_cast<float>(static_cast<std::int8_t>(ph[1 + i])) * mult_hi;
    if (residual) {
      v[i] += static_cast<float>(static_cast<std::int8_t>(pl[1 + i])) * mult_lo;
    }
  }
  *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

}  // namespace

int NpuDecodeFused(const std::uint8_t* const* hi, const std::uint8_t* const* lo,
                   float* out0, float* out1, float* out2, int tokens,
                   int slices, int per_tensor, std::size_t pitch, bool residual,
                   hipStream_t stream) {
  const int blocks = tokens * kSliceCols / 8 / kBlockGroups;
  NpuDecodeFusedKernel<<<dim3(blocks, slices), kBlockGroups, 0, stream>>>(
      hi, lo, out0, out1, out2, per_tensor, pitch, residual);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
