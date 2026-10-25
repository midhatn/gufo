#include "src/models/glm53_flash/kernels/rocm/embed.hpp"

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kHidden = 4096;
constexpr int kStreams = 4;
constexpr int kSub = kHidden / 16;

constexpr std::uint32_t kTypeQ8_0 = 8;
constexpr std::uint32_t kTypeQ6_K = 14;

__device__ float DecodeQ6K(const std::uint8_t* row, int e) {
  const std::uint8_t* blk = row + (e >> 8) * 210;
  const int r = e & 255;
  const int n = r >> 7;
  const int quad = (r >> 5) & 3;
  const int l = r & 31;
  const std::uint8_t* ql = blk + n * 64;
  const std::uint8_t* qh = blk + 128 + n * 32;
  const std::int8_t* sc =
      reinterpret_cast<const std::int8_t*>(blk + 192) + n * 8;
  const unsigned low = quad < 2 ? (ql[l + 32 * (quad & 1)] & 0xF)
                                : (ql[l + 32 * (quad & 1)] >> 4);
  const unsigned high = (qh[l] >> (2 * quad)) & 3U;
  const int q = static_cast<int>(low | (high << 4)) - 32;
  const _Float16 d = *reinterpret_cast<const _Float16*>(blk + 208);
  return static_cast<float>(d) * static_cast<float>(sc[(l >> 4) + 2 * quad]) *
         static_cast<float>(q);
}

template<typename Out>
__global__ __launch_bounds__(kSub) void EmbedKernel(
    const std::uint8_t* __restrict__ table, std::uint32_t type,
    std::size_t row_bytes, const std::int32_t* __restrict__ tokens,
    Out* __restrict__ out) {
  const std::uint8_t* row =
      table + static_cast<std::size_t>(tokens[blockIdx.x]) * row_bytes;
  Out* dst = out + static_cast<std::size_t>(blockIdx.x) * kStreams * kHidden;
  const int sub16 = threadIdx.x;
  float v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int e = sub16 * 16 + j;
    if (type == kTypeQ6_K) {
      v[j] = DecodeQ6K(row, e);
    } else if (type == kTypeQ8_0) {
      const std::uint8_t* blk = row + (e >> 5) * 34;
      v[j] = static_cast<float>(*reinterpret_cast<const _Float16*>(blk)) *
             static_cast<float>(
                 reinterpret_cast<const std::int8_t*>(blk + 2)[e & 31]);
    } else {
      v[j] = reinterpret_cast<const float*>(row)[e];
    }
  }
#pragma unroll
  for (int s = 0; s < kStreams; ++s) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      dst[s * kHidden + sub16 * 16 + j] = static_cast<Out>(v[j]);
    }
  }
}

}  // namespace

int EmbedTokens(const void* table, std::uint32_t type, std::size_t row_bytes,
                const std::int32_t* tokens, void* out, bool half, int count,
                hipStream_t stream) {
  const auto* base = static_cast<const std::uint8_t*>(table);
  if (half) {
    EmbedKernel<_Float16><<<count, kSub, 0, stream>>>(
        base, type, row_bytes, tokens, static_cast<_Float16*>(out));
  } else {
    EmbedKernel<float><<<count, kSub, 0, stream>>>(
        base, type, row_bytes, tokens, static_cast<float*>(out));
  }
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
