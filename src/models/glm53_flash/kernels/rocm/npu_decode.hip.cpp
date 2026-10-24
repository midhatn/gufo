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
    const std::uint8_t* const* __restrict__ streams, float* out0, float* out1,
    float* out2, int per_tensor, std::size_t pitch) {
  const int slice = blockIdx.y;
  const std::size_t g0 = std::size_t{blockIdx.x} * kBlockGroups;
  __shared__ __attribute__((aligned(16))) std::uint8_t stage[kStageBytes];
  const int t = threadIdx.x;
  if (t < kStageVec) {
    reinterpret_cast<uint4*>(stage)[t] =
        reinterpret_cast<const uint4*>(streams[slice] + g0 * 9)[t];
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
  const std::uint8_t* p = stage + gi * 9;
  const float mult = ldexpf(1.0F, static_cast<int>(p[0]) - 127) / 64.0F;
  const std::size_t row =
      tile_row * kTileRows + static_cast<std::size_t>((sr * 2 + br) * 8 + rib);
  const int col = tile_col * kTileCols + cc * 8;
  const int tensor = slice / per_tensor;
  float* out = tensor == 0 ? out0 : (tensor == 1 ? out1 : out2);
  float* dst = out + row * pitch + (slice % per_tensor) * kSliceCols + col;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = static_cast<float>(static_cast<std::int8_t>(p[1 + i])) * mult;
  }
  *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// Eight tokens of one 8-column group are consecutive 9-byte groups of the
// stream (nine aligned 64-bit words).
struct RunWords {
  unsigned long long w[9];
};

__device__ inline RunWords LoadRun(const std::uint8_t* stream, int row8,
                                   int col) {
  constexpr int kTilesN = kSliceCols / kTileCols;
  const int tile_row = row8 / kTileRows;
  const int r = row8 % kTileRows;
  const int tile_col = col / kTileCols;
  const int c = col % kTileCols;
  const int br = (r >> 3) & 1;
  const int sr = r >> 4;
  const int bc = (c >> 3) & 1;
  const int sc = c >> 4;
  const int gi = (bc << 3) | (br << 4) | (sc << 5) | (sr << 8);
  const auto* p = reinterpret_cast<const unsigned long long*>(
      stream + (static_cast<std::size_t>(tile_row * kTilesN + tile_col) *
                    kGroupsPerTile +
                gi) *
                   9);
  RunWords run;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    run.w[i] = p[i];
  }
  return run;
}

__device__ inline void DecodeRun(const RunWords& run, float (&v)[8][8]) {
  const auto byte_at = [&](int j) {
    return static_cast<std::uint32_t>((run.w[j >> 3] >> ((j & 7) * 8)) & 0xFFU);
  };
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float mult =
        ldexpf(1.0F, static_cast<int>(byte_at(9 * i)) - 127) / 64.0F;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      v[i][k] =
          static_cast<float>(static_cast<std::int8_t>(byte_at(9 * i + 1 + k))) *
          mult;
    }
  }
}

__device__ inline float Silu(float v) {
  return v * (1.0F / (1.0F + expf(-v)));
}

constexpr int kConvTokens = 64;
constexpr int kConvTaps = 4;
constexpr int kConvInner = 8192;

// One thread: eight adjacent channels of one tensor over kConvTokens tokens.
__global__ __launch_bounds__(256) void KdaConvStreamsKernel(
    const std::uint8_t* const* __restrict__ streams,
    const float* __restrict__ conv_q, const float* __restrict__ conv_k,
    const float* __restrict__ conv_v, const float* __restrict__ history,
    float* __restrict__ out_q, float* __restrict__ out_k,
    float* __restrict__ out_v, int tokens) {
  const int tensor = blockIdx.y;
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  const int c0 = group * 8;
  const int slice = c0 / kSliceCols;
  const int col = c0 % kSliceCols;
  const std::uint8_t* stream =
      streams[tensor * (kConvInner / kSliceCols) + slice];
  const float* weights = tensor == 0 ? conv_q : (tensor == 1 ? conv_k : conv_v);
  float* out = tensor == 0 ? out_q : (tensor == 1 ? out_k : out_v);
  float taps[8][kConvTaps];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 w4 =
        *reinterpret_cast<const float4*>(weights + (c0 + i) * kConvTaps);
    taps[i][0] = w4.x;
    taps[i][1] = w4.y;
    taps[i][2] = w4.z;
    taps[i][3] = w4.w;
  }
  const int t0 = blockIdx.z * kConvTokens;
  float window[3][8];
  if (t0 == 0) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        window[j][i] = history[(tensor * 3 + j) * kConvInner + c0 + i];
      }
    }
  } else {
    RunWords run = LoadRun(stream, t0 - 8, col);
    float v[8][8];
    DecodeRun(run, v);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        window[j][i] = v[5 + j][i];
      }
    }
  }
  for (int base = t0; base < min(t0 + kConvTokens, tokens); base += 8) {
    RunWords run = LoadRun(stream, base, col);
    float v[8][8];
    DecodeRun(run, v);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float acc = taps[i][3] * v[t][i] + taps[i][0] * window[0][i] +
                          taps[i][1] * window[1][i] + taps[i][2] * window[2][i];
        o[i] = Silu(acc);
        window[0][i] = window[1][i];
        window[1][i] = window[2][i];
        window[2][i] = v[t][i];
      }
      float* dst = out + static_cast<std::size_t>(base + t) * kConvInner + c0;
      *reinterpret_cast<float4*>(dst) = make_float4(o[0], o[1], o[2], o[3]);
      *reinterpret_cast<float4*>(dst + 4) = make_float4(o[4], o[5], o[6], o[7]);
    }
  }
}

// The history becomes the last three raw tokens of the chunk.
__global__ __launch_bounds__(256) void KdaConvHistoryKernel(
    const std::uint8_t* const* __restrict__ streams,
    float* __restrict__ history, int tokens) {
  const int tensor = blockIdx.y;
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  const int c0 = group * 8;
  const int slice = c0 / kSliceCols;
  const int col = c0 % kSliceCols;
  const std::uint8_t* stream =
      streams[tensor * (kConvInner / kSliceCols) + slice];
  RunWords run = LoadRun(stream, tokens - 8, col);
  float v[8][8];
  DecodeRun(run, v);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      history[(tensor * 3 + j) * kConvInner + c0 + i] = v[5 + j][i];
    }
  }
}

}  // namespace

int NpuDecodeFused(const std::uint8_t* const* streams, float* out0, float* out1,
                   float* out2, int tokens, int slices, int per_tensor,
                   std::size_t pitch, hipStream_t stream) {
  const int blocks = tokens * kSliceCols / 8 / kBlockGroups;
  NpuDecodeFusedKernel<<<dim3(blocks, slices), kBlockGroups, 0, stream>>>(
      streams, out0, out1, out2, per_tensor, pitch);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaConvStreams(const std::uint8_t* const* streams, const float* conv_q,
                   const float* conv_k, const float* conv_v, float* history,
                   float* out_q, float* out_k, float* out_v, int tokens,
                   hipStream_t stream) {
  constexpr int kThreads = 256;
  const dim3 grid(kConvInner / 8 / kThreads, 3,
                  (tokens + kConvTokens - 1) / kConvTokens);
  KdaConvStreamsKernel<<<grid, kThreads, 0, stream>>>(
      streams, conv_q, conv_k, conv_v, history, out_q, out_k, out_v, tokens);
  KdaConvHistoryKernel<<<dim3(kConvInner / 8 / kThreads, 3), kThreads, 0,
                         stream>>>(streams, history, tokens);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
