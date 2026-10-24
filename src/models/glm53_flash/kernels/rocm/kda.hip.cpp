#include "src/models/glm53_flash/kernels/rocm/kda.hpp"

#include "src/models/glm53_flash/kernels/rocm/xor_lane.hpp"

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

constexpr int kChunk = 16;
constexpr int kChunkThreads = kKdaHeadDim;

// Chunked form of KdaRecurKernel for prefill: kChunk tokens at a time, one
// block per head, one state row per thread. Within a chunk, with
// G_t = prod_{s<=t} decay_s, the sequential recurrence is
//   u_t = S0 (G_t k_t) + sum_{j<t} R_tj d_j,   d_t = beta_t (v_t - u_t)
//   o_t = S0 (G_t q_t) + sum_{j<=t} Q_tj d_j
//   S_n = S0 G_n + sum_j d_j (k_j G_n / G_j)^T
// where R_tj = (k_t G_t) . (k_j / G_j) and Q_tj = (q_t G_t) . (k_j / G_j).
// kChunk = 16 keeps G >= exp(-5 * 16) inside fp32. Padding tokens carry
// k = q = 0, decay = 1 and beta = 0, which leaves the state untouched.
__global__ __launch_bounds__(kChunkThreads) void KdaRecurChunkKernel(
    const float* __restrict__ qs, const float* __restrict__ ks,
    const float* __restrict__ vs, const float* __restrict__ decays,
    const float* __restrict__ betas, float* __restrict__ state,
    float* __restrict__ o, int tokens) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int wave = tid / kWaveLanes;
  const float out_scale = rsqrtf(static_cast<float>(kKdaHeadDim));
  __shared__ __attribute__((aligned(16))) float sKd[kChunk][kKdaHeadDim];
  __shared__ __attribute__((aligned(16))) float sQd[kChunk][kKdaHeadDim];
  __shared__ __attribute__((aligned(16))) float sKu[kChunk][kKdaHeadDim];
  __shared__ __attribute__((aligned(16))) float sKe[kChunk][kKdaHeadDim];
  __shared__ float sV[kChunk][kKdaHeadDim];
  __shared__ float sU[kChunk][kKdaHeadDim];
  __shared__ float sO[kChunk][kKdaHeadDim];
  __shared__ float sR[kChunk][kChunk];
  __shared__ float sQm[kChunk][kChunk];
  __shared__ float sBeta[kChunk];
  __shared__ __attribute__((aligned(16))) float sG[kKdaHeadDim];
  __shared__ float sNorm[2][kChunk][kKdaHeadDim / kWaveLanes];

  float s[kKdaHeadDim];
  const float* srow =
      state +
      (static_cast<std::size_t>(head) * kKdaHeadDim + tid) * kKdaHeadDim;
#pragma unroll
  for (int k = 0; k < kKdaHeadDim; k += 4) {
    const float4 v = *reinterpret_cast<const float4*>(srow + k);
    s[k] = v.x;
    s[k + 1] = v.y;
    s[k + 2] = v.z;
    s[k + 3] = v.w;
  }
  for (int c0 = 0; c0 < tokens; c0 += kChunk) {
    const int n = min(kChunk, tokens - c0);
    for (int t = 0; t < kChunk; ++t) {
      float qv = 0.0F;
      float kv = 0.0F;
      float vv = 0.0F;
      if (t < n) {
        const std::size_t base = static_cast<std::size_t>(c0 + t) * kKdaInner +
                                 head * kKdaHeadDim + tid;
        qv = qs[base];
        kv = ks[base];
        vv = vs[base];
      }
      sQd[t][tid] = qv;
      sKd[t][tid] = kv;
      sV[t][tid] = vv;
    }
    if (tid < kChunk) {
      sBeta[tid] =
          tid < n ? betas[static_cast<std::size_t>(c0 + tid) * kKdaHeads + head]
                  : 0.0F;
    }
    __syncthreads();
    for (int t = 0; t < kChunk; ++t) {
      const float qq = WaveSum(sQd[t][tid] * sQd[t][tid]);
      const float kk = WaveSum(sKd[t][tid] * sKd[t][tid]);
      if ((tid % kWaveLanes) == 0) {
        sNorm[0][t][wave] = qq;
        sNorm[1][t][wave] = kk;
      }
    }
    __syncthreads();
    float gam = 1.0F;
    for (int t = 0; t < kChunk; ++t) {
      const float qn = rsqrtf(((sNorm[0][t][0] + sNorm[0][t][1]) +
                               (sNorm[0][t][2] + sNorm[0][t][3])) +
                              kL2Eps);
      const float kn = rsqrtf(((sNorm[1][t][0] + sNorm[1][t][1]) +
                               (sNorm[1][t][2] + sNorm[1][t][3])) +
                              kL2Eps);
      const float a =
          t < n ? decays[static_cast<std::size_t>(c0 + t) * kKdaInner +
                         head * kKdaHeadDim + tid]
                : 1.0F;
      const float qv = sQd[t][tid] * qn;
      const float kv = sKd[t][tid] * kn;
      gam *= a;
      sQd[t][tid] = qv * gam;
      sKd[t][tid] = kv * gam;
      sKu[t][tid] = kv / gam;
    }
    sG[tid] = gam;
    for (int t = 0; t < kChunk; ++t) {
      sKe[t][tid] = sKu[t][tid] * gam;
    }
    __syncthreads();
    for (int p = tid; p < kChunk * kChunk; p += kChunkThreads) {
      const int t = p / kChunk;
      const int j = p % kChunk;
      float dr = 0.0F;
      float dq = 0.0F;
      if (j <= t) {
#pragma unroll 8
        for (int e = 0; e < kKdaHeadDim; e += 4) {
          const float4 ku = *reinterpret_cast<const float4*>(&sKu[j][e]);
          const float4 kd = *reinterpret_cast<const float4*>(&sKd[t][e]);
          const float4 qd = *reinterpret_cast<const float4*>(&sQd[t][e]);
          dr += kd.x * ku.x + kd.y * ku.y + kd.z * ku.z + kd.w * ku.w;
          dq += qd.x * ku.x + qd.y * ku.y + qd.z * ku.z + qd.w * ku.w;
        }
      }
      sR[t][j] = j < t ? dr : 0.0F;
      sQm[t][j] = dq;
    }
    __syncthreads();
    for (int c = 0; c < kChunk; ++c) {
      float u = 0.0F;
      float oo = 0.0F;
#pragma unroll
      for (int k = 0; k < kKdaHeadDim; k += 4) {
        const float4 kd = *reinterpret_cast<const float4*>(&sKd[c][k]);
        const float4 qd = *reinterpret_cast<const float4*>(&sQd[c][k]);
        u += s[k] * kd.x + s[k + 1] * kd.y + s[k + 2] * kd.z + s[k + 3] * kd.w;
        oo += s[k] * qd.x + s[k + 1] * qd.y + s[k + 2] * qd.z + s[k + 3] * qd.w;
      }
      sU[c][tid] = u;
      sO[c][tid] = oo;
    }
    float delta[kChunk];
#pragma unroll
    for (int t = 0; t < kChunk; ++t) {
      float acc = sU[t][tid];
      float out = sO[t][tid];
#pragma unroll
      for (int j = 0; j < t; ++j) {
        acc += sR[t][j] * delta[j];
      }
      delta[t] = sBeta[t] * (sV[t][tid] - acc);
#pragma unroll
      for (int j = 0; j <= t; ++j) {
        out += sQm[t][j] * delta[j];
      }
      if (t < n) {
        o[static_cast<std::size_t>(c0 + t) * kKdaInner + head * kKdaHeadDim +
          tid] = out * out_scale;
      }
    }
#pragma unroll
    for (int k = 0; k < kKdaHeadDim; k += 4) {
      const float4 g = *reinterpret_cast<const float4*>(&sG[k]);
      s[k] *= g.x;
      s[k + 1] *= g.y;
      s[k + 2] *= g.z;
      s[k + 3] *= g.w;
#pragma unroll
      for (int j = 0; j < kChunk; ++j) {
        const float4 ke = *reinterpret_cast<const float4*>(&sKe[j][k]);
        s[k] += delta[j] * ke.x;
        s[k + 1] += delta[j] * ke.y;
        s[k + 2] += delta[j] * ke.z;
        s[k + 3] += delta[j] * ke.w;
      }
    }
    __syncthreads();
  }
  float* drow = state + (static_cast<std::size_t>(head) * kKdaHeadDim + tid) *
                            kKdaHeadDim;
#pragma unroll
  for (int k = 0; k < kKdaHeadDim; k += 4) {
    *reinterpret_cast<float4*>(drow + k) =
        make_float4(s[k], s[k + 1], s[k + 2], s[k + 3]);
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

int KdaRecurChunked(const float* q, const float* k, const float* v,
                    const float* decay, const float* beta, float* state,
                    float* o, hipStream_t stream, int tokens) {
  KdaRecurChunkKernel<<<kKdaHeads, kChunkThreads, 0, stream>>>(
      q, k, v, decay, beta, state, o, tokens);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int KdaOut(float* o, const float* norm_w, const float* g, hipStream_t stream,
           int tokens) {
  KdaOutKernel<<<kKdaHeads * tokens, kKdaHeadDim, 0, stream>>>(o, norm_w, g);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
