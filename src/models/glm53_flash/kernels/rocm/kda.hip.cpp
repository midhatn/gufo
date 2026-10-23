#include "src/models/glm53_flash/kernels/rocm/kda.hpp"

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kWaveLanes = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kRowsPerWave = 8;
constexpr int kRowsPerBlock = kWavesPerBlock * kRowsPerWave;
constexpr int kLaneElems = kKdaHeadDim / kWaveLanes;
constexpr float kL2Eps = 1e-6F;
constexpr float kRmsEps = 1e-5F;
constexpr float kDecayScale = 5.0F;
constexpr int kElemThreads = 256;

__device__ float Sigmoid(float v) {
  return 1.0F / (1.0F + expf(-v));
}

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
  return __shfl_xor(v, offset, kWaveLanes);
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}

__device__ float WaveSum(float v) {
  for (int offset = kWaveLanes / 2; offset > 0; offset >>= 1) {
    v += XorLane(v, offset);
  }
  return v;
}

__global__ void KdaConvKernel(float* __restrict__ q, float* __restrict__ k,
                              float* __restrict__ v, float* __restrict__ hist,
                              const float* __restrict__ cq,
                              const float* __restrict__ ck,
                              const float* __restrict__ cv, int tokens) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int stream = blockIdx.y;
  if (ch >= kKdaInner) {
    return;
  }
  float* data = stream == 0 ? q : (stream == 1 ? k : v);
  const float* cw = stream == 0 ? cq : (stream == 1 ? ck : cv);
  const float* tap_w = cw + ch * kKdaConvTaps;
  float* h = hist + stream * 3 * kKdaInner;
  float h0 = h[0 * kKdaInner + ch];
  float h1 = h[1 * kKdaInner + ch];
  float h2 = h[2 * kKdaInner + ch];
  for (int t = 0; t < tokens; ++t) {
    float* p = data + static_cast<std::size_t>(t) * kKdaInner + ch;
    const float raw = *p;
    const float acc =
        tap_w[3] * raw + tap_w[0] * h0 + tap_w[1] * h1 + tap_w[2] * h2;
    *p = acc * Sigmoid(acc);
    h0 = h1;
    h1 = h2;
    h2 = raw;
  }
  h[0 * kKdaInner + ch] = h0;
  h[1 * kKdaInner + ch] = h1;
  h[2 * kKdaInner + ch] = h2;
}

__global__ void KdaGatesKernel(const float* __restrict__ f,
                               const float* __restrict__ dt,
                               const float* __restrict__ ssm_a,
                               const float* __restrict__ braw,
                               float* __restrict__ decay,
                               float* __restrict__ beta, int tokens) {
  const long long i =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < static_cast<long long>(tokens) * kKdaInner) {
    const int channel = static_cast<int>(i % kKdaInner);
    const float a = ssm_a[channel / kKdaHeadDim];
    decay[i] = expf(-kDecayScale / (1.0F + expf((f[i] + dt[channel]) * a)));
  }
  if (i < static_cast<long long>(tokens) * kKdaHeads) {
    beta[i] = Sigmoid(braw[i]);
  }
}

__global__ __launch_bounds__(kWavesPerBlock* kWaveLanes) void KdaRecurKernel(
    const float* __restrict__ qs, const float* __restrict__ ks,
    const float* __restrict__ vs, const float* __restrict__ decays,
    const float* __restrict__ betas, float* __restrict__ state,
    float* __restrict__ o, int tokens) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x % kWaveLanes;
  const int wave = threadIdx.x / kWaveLanes;
  const float out_scale = rsqrtf(static_cast<float>(kKdaHeadDim));

  float s[kRowsPerWave][kLaneElems];
  for (int r = 0; r < kRowsPerWave; ++r) {
    const int row = blockIdx.y * kRowsPerBlock + wave * kRowsPerWave + r;
    const float* srow =
        state +
        (static_cast<std::size_t>(head) * kKdaHeadDim + row) * kKdaHeadDim;
    for (int e = 0; e < kLaneElems; ++e) {
      s[r][e] = srow[lane * kLaneElems + e];
    }
  }

  for (int t = 0; t < tokens; ++t) {
    const std::size_t base =
        static_cast<std::size_t>(t) * kKdaInner + head * kKdaHeadDim;
    float q[kLaneElems];
    float k[kLaneElems];
    float dec[kLaneElems];
    float qq = 0.0F;
    float kk = 0.0F;
    for (int e = 0; e < kLaneElems; ++e) {
      const int i = lane * kLaneElems + e;
      q[e] = qs[base + i];
      k[e] = ks[base + i];
      dec[e] = decays[base + i];
      qq += q[e] * q[e];
      kk += k[e] * k[e];
    }
    const float qn = rsqrtf(WaveSum(qq) + kL2Eps);
    const float kn = rsqrtf(WaveSum(kk) + kL2Eps);
    for (int e = 0; e < kLaneElems; ++e) {
      q[e] *= qn;
      k[e] *= kn;
    }
    const float b = betas[static_cast<std::size_t>(t) * kKdaHeads + head];
    for (int r = 0; r < kRowsPerWave; ++r) {
      const int row = blockIdx.y * kRowsPerBlock + wave * kRowsPerWave + r;
      float dot_k = 0.0F;
      for (int e = 0; e < kLaneElems; ++e) {
        s[r][e] *= dec[e];
        dot_k += s[r][e] * k[e];
      }
      const float u = WaveSum(dot_k);
      const float delta = (vs[base + row] - u) * b;
      float dot_q = 0.0F;
      for (int e = 0; e < kLaneElems; ++e) {
        s[r][e] += delta * k[e];
        dot_q += s[r][e] * q[e];
      }
      const float out = WaveSum(dot_q);
      if (lane == 0) {
        o[base + row] = out * out_scale;
      }
    }
  }

  for (int r = 0; r < kRowsPerWave; ++r) {
    const int row = blockIdx.y * kRowsPerBlock + wave * kRowsPerWave + r;
    float* srow = state + (static_cast<std::size_t>(head) * kKdaHeadDim + row) *
                              kKdaHeadDim;
    for (int e = 0; e < kLaneElems; ++e) {
      srow[lane * kLaneElems + e] = s[r][e];
    }
  }
}

__global__ __launch_bounds__(kKdaHeadDim) void KdaOutKernel(
    float* __restrict__ o, const float* __restrict__ norm_w,
    const float* __restrict__ g) {
  __shared__ float scratch[kKdaHeadDim / kWaveLanes];
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * kKdaHeadDim;
  const int i = threadIdx.x;
  const float v = o[base + i];
  float sq = WaveSum(v * v);
  if ((i % kWaveLanes) == 0) {
    scratch[i / kWaveLanes] = sq;
  }
  __syncthreads();
  float total = 0.0F;
  for (int w = 0; w < kKdaHeadDim / kWaveLanes; ++w) {
    total += scratch[w];
  }
  const float inv = rsqrtf(total / kKdaHeadDim + kRmsEps);
  o[base + i] = v * inv * norm_w[i] * Sigmoid(g[base + i]);
}

}  // namespace

int KdaConv(float* q, float* k, float* v, float* hist, const float* conv_q,
            const float* conv_k, const float* conv_v, hipStream_t stream,
            int tokens) {
  KdaConvKernel<<<dim3((kKdaInner + kElemThreads - 1) / kElemThreads, 3),
                  kElemThreads, 0, stream>>>(q, k, v, hist, conv_q, conv_k,
                                             conv_v, tokens);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaGates(const float* f, const float* dt, const float* ssm_a,
             const float* braw, float* decay, float* beta, hipStream_t stream,
             int tokens) {
  const long long total = static_cast<long long>(tokens) * kKdaInner;
  KdaGatesKernel<<<static_cast<unsigned>((total + kElemThreads - 1) /
                                         kElemThreads),
                   kElemThreads, 0, stream>>>(f, dt, ssm_a, braw, decay, beta,
                                              tokens);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaRecur(const float* q, const float* k, const float* v, const float* decay,
             const float* beta, float* state, float* o, hipStream_t stream,
             int tokens) {
  KdaRecurKernel<<<dim3(kKdaHeads, kKdaHeadDim / kRowsPerBlock),
                   kWavesPerBlock * kWaveLanes, 0, stream>>>(
      q, k, v, decay, beta, state, o, tokens);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaOut(float* o, const float* norm_w, const float* g, hipStream_t stream,
           int tokens) {
  KdaOutKernel<<<kKdaHeads * tokens, kKdaHeadDim, 0, stream>>>(o, norm_w, g);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
