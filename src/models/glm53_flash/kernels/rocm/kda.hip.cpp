#include "src/models/glm53_flash/kernels/rocm/kda.hpp"

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kWaveLanes = 32;
constexpr int kWavesPerBlock = 8;
constexpr int kRowsPerWave = 4;
constexpr int kRowsPerBlock = kWavesPerBlock * kRowsPerWave;
constexpr int kLaneElems = kKdaHeadDim / kWaveLanes;
constexpr float kL2Eps = 1e-6F;
constexpr float kRmsEps = 1e-5F;
constexpr float kDecayScale = 5.0F;
constexpr int kElemThreads = 256;

__device__ float Sigmoid(float v) {
  return 1.0F / (1.0F + expf(-v));
}

__device__ float WaveSum(float v) {
  for (int offset = kWaveLanes / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWaveLanes);
  }
  return v;
}

__global__ void KdaConvKernel(float* __restrict__ qkv, float* __restrict__ hist,
                              const float* __restrict__ cq,
                              const float* __restrict__ ck,
                              const float* __restrict__ cv) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  const int stream = blockIdx.y;
  if (ch >= kKdaInner) {
    return;
  }
  const float* cw = stream == 0 ? cq : (stream == 1 ? ck : cv);
  const float* tap_w = cw + ch * kKdaConvTaps;
  float* h = hist + stream * 3 * kKdaInner;
  const float raw = qkv[stream * kKdaInner + ch];
  float acc = tap_w[3] * raw;
  for (int tap = 0; tap < 3; ++tap) {
    acc += tap_w[tap] * h[tap * kKdaInner + ch];
  }
  qkv[stream * kKdaInner + ch] = acc * Sigmoid(acc);
  h[0 * kKdaInner + ch] = h[1 * kKdaInner + ch];
  h[1 * kKdaInner + ch] = h[2 * kKdaInner + ch];
  h[2 * kKdaInner + ch] = raw;
}

__global__ void KdaGatesKernel(const float* __restrict__ f,
                               const float* __restrict__ dt,
                               const float* __restrict__ ssm_a,
                               const float* __restrict__ braw,
                               float* __restrict__ decay,
                               float* __restrict__ beta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kKdaInner) {
    const float a = ssm_a[i / kKdaHeadDim];
    decay[i] = expf(-kDecayScale / (1.0F + expf((f[i] + dt[i]) * a)));
  }
  if (i < kKdaHeads) {
    beta[i] = Sigmoid(braw[i]);
  }
}

__global__ __launch_bounds__(kWavesPerBlock* kWaveLanes) void KdaRecurKernel(
    const float* __restrict__ qkv, const float* __restrict__ decay,
    const float* __restrict__ beta, float* __restrict__ state,
    float* __restrict__ o) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x % kWaveLanes;
  const int wave = threadIdx.x / kWaveLanes;
  const float* qh = qkv + head * kKdaHeadDim;
  const float* kh = qkv + kKdaInner + head * kKdaHeadDim;
  const float* vh = qkv + 2 * kKdaInner + head * kKdaHeadDim;

  float q[kLaneElems];
  float k[kLaneElems];
  float dec[kLaneElems];
  float qq = 0.0F;
  float kk = 0.0F;
  for (int e = 0; e < kLaneElems; ++e) {
    const int i = lane * kLaneElems + e;
    q[e] = qh[i];
    k[e] = kh[i];
    dec[e] = decay[head * kKdaHeadDim + i];
    qq += q[e] * q[e];
    kk += k[e] * k[e];
  }
  const float qn = rsqrtf(WaveSum(qq) + kL2Eps);
  const float kn = rsqrtf(WaveSum(kk) + kL2Eps);
  for (int e = 0; e < kLaneElems; ++e) {
    q[e] *= qn;
    k[e] *= kn;
  }
  const float b = beta[head];
  const float out_scale = rsqrtf(static_cast<float>(kKdaHeadDim));

  for (int r = 0; r < kRowsPerWave; ++r) {
    const int row = blockIdx.y * kRowsPerBlock + wave * kRowsPerWave + r;
    float* srow = state + (static_cast<std::size_t>(head) * kKdaHeadDim + row) *
                              kKdaHeadDim;
    float s[kLaneElems];
    float dot_k = 0.0F;
    for (int e = 0; e < kLaneElems; ++e) {
      s[e] = srow[lane * kLaneElems + e] * dec[e];
      dot_k += s[e] * k[e];
    }
    const float u = WaveSum(dot_k);
    const float delta = (vh[row] - u) * b;
    float dot_q = 0.0F;
    for (int e = 0; e < kLaneElems; ++e) {
      s[e] += delta * k[e];
      srow[lane * kLaneElems + e] = s[e];
      dot_q += s[e] * q[e];
    }
    const float out = WaveSum(dot_q);
    if (lane == 0) {
      o[head * kKdaHeadDim + row] = out * out_scale;
    }
  }
}

__global__ __launch_bounds__(kKdaHeadDim) void KdaOutKernel(
    float* __restrict__ o, const float* __restrict__ norm_w,
    const float* __restrict__ g) {
  __shared__ float scratch[kKdaHeadDim / kWaveLanes];
  const int head = blockIdx.x;
  const int i = threadIdx.x;
  const float v = o[head * kKdaHeadDim + i];
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
  o[head * kKdaHeadDim + i] =
      v * inv * norm_w[i] * Sigmoid(g[head * kKdaHeadDim + i]);
}

}  // namespace

int KdaConv(float* qkv, float* hist, const float* conv_q, const float* conv_k,
            const float* conv_v, hipStream_t stream) {
  KdaConvKernel<<<dim3((kKdaInner + kElemThreads - 1) / kElemThreads, 3),
                  kElemThreads, 0, stream>>>(qkv, hist, conv_q, conv_k, conv_v);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaGates(const float* f, const float* dt, const float* ssm_a,
             const float* braw, float* decay, float* beta, hipStream_t stream) {
  KdaGatesKernel<<<(kKdaInner + kElemThreads - 1) / kElemThreads, kElemThreads,
                   0, stream>>>(f, dt, ssm_a, braw, decay, beta);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaRecur(const float* qkv, const float* decay, const float* beta,
             float* state, float* o, hipStream_t stream) {
  KdaRecurKernel<<<dim3(kKdaHeads, kKdaHeadDim / kRowsPerBlock),
                   kWavesPerBlock * kWaveLanes, 0, stream>>>(qkv, decay, beta,
                                                             state, o);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaOut(float* o, const float* norm_w, const float* g, hipStream_t stream) {
  KdaOutKernel<<<kKdaHeads, kKdaHeadDim, 0, stream>>>(o, norm_w, g);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
