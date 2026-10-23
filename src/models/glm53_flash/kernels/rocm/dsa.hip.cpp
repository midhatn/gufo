#include "src/models/glm53_flash/kernels/rocm/dsa.hpp"

#include <cstdint>

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kThreads = 1024;
constexpr int kWaveLanes = 32;
constexpr int kMaxWaves = kThreads / kWaveLanes;
constexpr int kRadixBins = 256;
constexpr int kRadixPasses = 4;
constexpr int kScoreWaves = 8;
constexpr int kAttendThreads = 256;
constexpr int kAttendWaves = kAttendThreads / kWaveLanes;
constexpr int kIndexElems = kDsaIndexDim / kWaveLanes;

__device__ float WaveSum(float v) {
  for (int offset = kWaveLanes / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset, kWaveLanes);
  }
  return v;
}

__device__ float WaveMax(float v) {
  for (int offset = kWaveLanes / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset, kWaveLanes));
  }
  return v;
}

__device__ float BlockSum(float v, float* scratch) {
  v = WaveSum(v);
  const int wave = threadIdx.x / kWaveLanes;
  const int waves = blockDim.x / kWaveLanes;
  __syncthreads();
  if ((threadIdx.x % kWaveLanes) == 0) {
    scratch[wave] = v;
  }
  __syncthreads();
  float total = 0.0F;
  for (int w = 0; w < waves; ++w) {
    total += scratch[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* scratch) {
  v = WaveMax(v);
  const int wave = threadIdx.x / kWaveLanes;
  const int waves = blockDim.x / kWaveLanes;
  __syncthreads();
  if ((threadIdx.x % kWaveLanes) == 0) {
    scratch[wave] = v;
  }
  __syncthreads();
  float best = scratch[0];
  for (int w = 1; w < waves; ++w) {
    best = fmaxf(best, scratch[w]);
  }
  return best;
}

__global__ __launch_bounds__(kThreads) void RmsNormKernel(
    float* __restrict__ x, const float* __restrict__ w, int n, float eps) {
  __shared__ float scratch[kMaxWaves];
  float sq = 0.0F;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    sq += x[i] * x[i];
  }
  const float inv = rsqrtf(BlockSum(sq, scratch) / n + eps);
  for (int i = threadIdx.x; i < n; i += kThreads) {
    x[i] = x[i] * inv * w[i];
  }
}

__global__ __launch_bounds__(kThreads) void LayerNormKernel(
    float* __restrict__ x, const float* __restrict__ w,
    const float* __restrict__ b, int n, float eps) {
  __shared__ float scratch[kMaxWaves];
  float sum = 0.0F;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    sum += x[i];
  }
  const float mu = BlockSum(sum, scratch) / n;
  float sq = 0.0F;
  for (int i = threadIdx.x; i < n; i += kThreads) {
    const float d = x[i] - mu;
    sq += d * d;
  }
  const float inv = rsqrtf(BlockSum(sq, scratch) / n + eps);
  for (int i = threadIdx.x; i < n; i += kThreads) {
    x[i] = (x[i] - mu) * inv * w[i] + b[i];
  }
}

__global__ void DsaAppendKernel(DsaCache cache, const float* __restrict__ lat,
                                const float* __restrict__ ik,
                                const float* __restrict__ gate,
                                const float* __restrict__ ape, int pos) {
  const int d = threadIdx.x;
  for (int c = d; c < kDsaLatent; c += blockDim.x) {
    cache.lat[static_cast<std::size_t>(pos) * kDsaLatent + c] = lat[c];
  }
  cache.ik[static_cast<std::size_t>(pos) * kDsaIndexDim + d] = ik[d];
  cache.gate[static_cast<std::size_t>(pos) * kDsaIndexDim + d] = gate[d];
  if (((pos + 1) % kDsaPool) != 0) {
    return;
  }
  __syncthreads();
  const int base = pos + 1 - kDsaPool;
  float x4[kDsaPool];
  float mx = -1e30F;
  for (int s = 0; s < kDsaPool; ++s) {
    x4[s] = cache.gate[static_cast<std::size_t>(base + s) * kDsaIndexDim + d] +
            ape[d + kDsaIndexDim * s];
    mx = fmaxf(mx, x4[s]);
  }
  float sum = 0.0F;
  for (int s = 0; s < kDsaPool; ++s) {
    x4[s] = expf(x4[s] - mx);
    sum += x4[s];
  }
  float acc = 0.0F;
  for (int s = 0; s < kDsaPool; ++s) {
    acc += (x4[s] / sum) *
           cache.ik[static_cast<std::size_t>(base + s) * kDsaIndexDim + d];
  }
  cache.pooled[static_cast<std::size_t>(base / kDsaPool) * kDsaIndexDim + d] =
      acc;
}

__global__ __launch_bounds__(kScoreWaves* kWaveLanes) void DsaScoresKernel(
    const float* __restrict__ iq, const float* __restrict__ proj,
    const float* __restrict__ pooled, int npool, float* __restrict__ scores) {
  __shared__ float q[kDsaIndexHeads * kDsaIndexDim];
  __shared__ float w[kDsaIndexHeads];
  for (int i = threadIdx.x; i < kDsaIndexHeads * kDsaIndexDim;
       i += blockDim.x) {
    q[i] = iq[i];
  }
  if (threadIdx.x < kDsaIndexHeads) {
    w[threadIdx.x] = proj[threadIdx.x] *
                     rsqrtf(static_cast<float>(kDsaIndexDim * kDsaIndexHeads));
  }
  __syncthreads();
  const int lane = threadIdx.x % kWaveLanes;
  const int pool = blockIdx.x * kScoreWaves + threadIdx.x / kWaveLanes;
  if (pool >= npool) {
    return;
  }
  float kp[kIndexElems];
  for (int e = 0; e < kIndexElems; ++e) {
    kp[e] = pooled[static_cast<std::size_t>(pool) * kDsaIndexDim +
                   lane * kIndexElems + e];
  }
  float total = 0.0F;
  for (int h = 0; h < kDsaIndexHeads; ++h) {
    float dot = 0.0F;
    for (int e = 0; e < kIndexElems; ++e) {
      dot += q[h * kDsaIndexDim + lane * kIndexElems + e] * kp[e];
    }
    dot = WaveSum(dot);
    total += fmaxf(dot, 0.0F) * w[h];
  }
  if (lane == 0) {
    scores[pool] = total;
  }
}

__device__ std::uint32_t OrderedKey(float v) {
  const std::uint32_t u = __float_as_uint(v);
  return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}

// Exclusive scan of one int per thread across the block; returns the total.
__device__ int BlockExclusiveScan(int v, int* scratch, int* exclusive) {
  scratch[threadIdx.x] = v;
  __syncthreads();
  for (int offset = 1; offset < kThreads; offset <<= 1) {
    int add = 0;
    if (threadIdx.x >= offset) {
      add = scratch[threadIdx.x - offset];
    }
    __syncthreads();
    scratch[threadIdx.x] += add;
    __syncthreads();
  }
  *exclusive = scratch[threadIdx.x] - v;
  const int total = scratch[kThreads - 1];
  __syncthreads();
  return total;
}

__global__ __launch_bounds__(kThreads) void DsaSelectKernel(
    const float* __restrict__ scores, int npool, int top,
    int* __restrict__ selected) {
  __shared__ std::uint32_t hist[kRadixBins];
  __shared__ std::uint32_t prefix;
  __shared__ int remaining;
  __shared__ int scan[kThreads];
  __shared__ int carry[2];
  const int k = npool < top ? npool : top;
  if (k == npool) {
    for (int i = threadIdx.x; i < npool; i += kThreads) {
      selected[i] = i;
    }
    return;
  }
  if (threadIdx.x == 0) {
    prefix = 0;
    remaining = k;
  }
  __syncthreads();
  for (int pass = 0; pass < kRadixPasses; ++pass) {
    const int shift = 24 - 8 * pass;
    for (int b = threadIdx.x; b < kRadixBins; b += kThreads) {
      hist[b] = 0;
    }
    __syncthreads();
    const std::uint32_t high_mask =
        pass == 0 ? 0U : (0xFFFFFFFFU << (shift + 8));
    for (int i = threadIdx.x; i < npool; i += kThreads) {
      const std::uint32_t key = OrderedKey(scores[i]);
      if ((key & high_mask) == prefix) {
        atomicAdd(&hist[(key >> shift) & 0xFFU], 1U);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int need = remaining;
      int cum = 0;
      int bin = kRadixBins - 1;
      for (; bin > 0; --bin) {
        if (cum + static_cast<int>(hist[bin]) >= need) {
          break;
        }
        cum += static_cast<int>(hist[bin]);
      }
      remaining = need - cum;
      prefix |= static_cast<std::uint32_t>(bin) << shift;
    }
    __syncthreads();
  }
  const std::uint32_t threshold = prefix;
  const int need_equal = remaining;
  if (threadIdx.x == 0) {
    carry[0] = 0;
    carry[1] = 0;
  }
  __syncthreads();
  for (int base = 0; base < npool; base += kThreads) {
    const int i = base + threadIdx.x;
    const std::uint32_t key = i < npool ? OrderedKey(scores[i]) : 0U;
    const bool greater = i < npool && key > threshold;
    const bool equal = i < npool && key == threshold;
    int eq_excl = 0;
    const int eq_total = BlockExclusiveScan(equal ? 1 : 0, scan, &eq_excl);
    const bool take = greater || (equal && (carry[0] + eq_excl) < need_equal);
    int out_excl = 0;
    const int out_total = BlockExclusiveScan(take ? 1 : 0, scan, &out_excl);
    if (take) {
      selected[carry[1] + out_excl] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry[0] += eq_total;
      carry[1] += out_total;
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kAttendThreads) void DsaAttendKernel(
    const float* __restrict__ qabs, const float* __restrict__ lat,
    const int* __restrict__ selected, int nselected, int pos, float scale,
    float* __restrict__ ctx) {
  __shared__ float q[kDsaLatent];
  __shared__ float prob[kDsaMaxVisible];
  __shared__ int where[kDsaMaxVisible];
  __shared__ float scratch[kAttendWaves];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid % kWaveLanes;
  const int wave = tid / kWaveLanes;
  const int tail = (pos + 1) % kDsaPool;
  const int nvis = nselected * kDsaPool + tail;

  for (int c = tid; c < kDsaLatent; c += kAttendThreads) {
    q[c] = qabs[head * kDsaLatent + c];
  }
  for (int j = tid; j < nvis; j += kAttendThreads) {
    where[j] = j < nselected * kDsaPool
                   ? selected[j / kDsaPool] * kDsaPool + (j % kDsaPool)
                   : pos - (j - nselected * kDsaPool);
  }
  __syncthreads();

  for (int j = wave; j < nvis; j += kAttendWaves) {
    const float* row = lat + static_cast<std::size_t>(where[j]) * kDsaLatent;
    float dot = 0.0F;
    for (int c = lane; c < kDsaLatent; c += kWaveLanes) {
      dot += q[c] * row[c];
    }
    dot = WaveSum(dot) * scale;
    if (lane == 0) {
      prob[j] = dot;
    }
  }
  __syncthreads();
  float best = -1e30F;
  for (int j = tid; j < nvis; j += kAttendThreads) {
    best = fmaxf(best, prob[j]);
  }
  best = BlockMax(best, scratch);
  float denom = 0.0F;
  for (int j = tid; j < nvis; j += kAttendThreads) {
    const float e = expf(prob[j] - best);
    prob[j] = e;
    denom += e;
  }
  denom = BlockSum(denom, scratch);
  __syncthreads();
  const float inv = 1.0F / denom;
  for (int c = tid; c < kDsaLatent; c += kAttendThreads) {
    float acc = 0.0F;
    for (int j = 0; j < nvis; ++j) {
      acc += prob[j] * inv *
             lat[static_cast<std::size_t>(where[j]) * kDsaLatent + c];
    }
    ctx[head * kDsaLatent + c] = acc;
  }
}

}  // namespace

int RmsNorm(float* x, const float* w, int n, float eps, hipStream_t stream) {
  RmsNormKernel<<<1, kThreads, 0, stream>>>(x, w, n, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int LayerNorm(float* x, const float* w, const float* b, int n, float eps,
              hipStream_t stream) {
  LayerNormKernel<<<1, kThreads, 0, stream>>>(x, w, b, n, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAppend(const DsaCache& cache, const float* lat, const float* ik,
              const float* gate, const float* ape, int pos,
              hipStream_t stream) {
  DsaAppendKernel<<<1, kDsaIndexDim, 0, stream>>>(cache, lat, ik, gate, ape,
                                                  pos);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaScores(const float* iq, const float* proj, const float* pooled,
              int npool, float* scores, hipStream_t stream) {
  if (npool <= 0) {
    return 0;
  }
  DsaScoresKernel<<<(npool + kScoreWaves - 1) / kScoreWaves,
                    kScoreWaves * kWaveLanes, 0, stream>>>(iq, proj, pooled,
                                                           npool, scores);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaSelect(const float* scores, int npool, int top_pools, int* selected,
              hipStream_t stream) {
  if (npool <= 0) {
    return 0;
  }
  DsaSelectKernel<<<1, kThreads, 0, stream>>>(scores, npool, top_pools,
                                              selected);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAttend(const float* qabs, const float* lat, const int* selected,
              int nselected, int pos, float scale, float* ctx,
              hipStream_t stream) {
  DsaAttendKernel<<<kDsaHeads, kAttendThreads, 0, stream>>>(
      qabs, lat, selected, nselected, pos, scale, ctx);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
