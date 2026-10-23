#include "src/models/glm53_flash/kernels/rocm/dsa.hpp"

#include <cstdint>

#include "src/models/glm53_flash/kernels/rocm/xor_lane.hpp"

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
  x += static_cast<std::size_t>(blockIdx.x) * n;
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
  x += static_cast<std::size_t>(blockIdx.x) * n;
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
                                const float* __restrict__ ape, int pos,
                                const int* __restrict__ pos_dev) {
  if (pos_dev != nullptr) {
    pos = *pos_dev;
  }
  const int d = threadIdx.x;
  for (int c = d; c < kDsaLatent; c += blockDim.x) {
    cache.lat[static_cast<std::size_t>(pos) * kDsaLatent + c] = lat[c];
    reinterpret_cast<_Float16*>(
        cache.lat_half)[static_cast<std::size_t>(pos) * kDsaLatent + c] =
        static_cast<_Float16>(lat[c]);
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

// Rows of `tokens` consecutive tokens starting at position pos0.
__global__ void DsaAppendRowsKernel(DsaCache cache,
                                    const float* __restrict__ lat,
                                    const float* __restrict__ ik,
                                    const float* __restrict__ gate, int pos0) {
  const int token = blockIdx.x;
  const std::size_t pos = static_cast<std::size_t>(pos0) + token;
  for (int c = threadIdx.x; c < kDsaLatent; c += blockDim.x) {
    const float v = lat[static_cast<std::size_t>(token) * kDsaLatent + c];
    cache.lat[pos * kDsaLatent + c] = v;
    reinterpret_cast<_Float16*>(cache.lat_half)[pos * kDsaLatent + c] =
        static_cast<_Float16>(v);
  }
  cache.ik[pos * kDsaIndexDim + threadIdx.x] =
      ik[static_cast<std::size_t>(token) * kDsaIndexDim + threadIdx.x];
  cache.gate[pos * kDsaIndexDim + threadIdx.x] =
      gate[static_cast<std::size_t>(token) * kDsaIndexDim + threadIdx.x];
}

// One block per pool completed by tokens pos0 .. pos0 + tokens - 1.
__global__ void DsaPoolKernel(DsaCache cache, const float* __restrict__ ape,
                              int first_pool) {
  const int d = threadIdx.x;
  const int pool = first_pool + blockIdx.x;
  const int base = pool * kDsaPool;
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
  cache.pooled[static_cast<std::size_t>(pool) * kDsaIndexDim + d] = acc;
}

// out[h][t][i] = in[t][h][i] for `heads` slabs of `width` floats.
__global__ void HeadMajorKernel(const float* __restrict__ in,
                                _Float16* __restrict__ out, int tokens,
                                int heads, int width) {
  const long long n = static_cast<long long>(tokens) * heads * width;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int w = static_cast<int>(i % width);
    const long long th = i / width;
    const int h = static_cast<int>(th % heads);
    const int t = static_cast<int>(th / heads);
    out[(static_cast<std::size_t>(h) * tokens + t) * width + w] =
        static_cast<_Float16>(in[i]);
  }
}

// out[t][h][i] = in[h][t][i].
__global__ void TokenMajorKernel(const _Float16* __restrict__ in,
                                 float* __restrict__ out, int tokens, int heads,
                                 int width) {
  const long long n = static_cast<long long>(tokens) * heads * width;
  for (long long i =
           static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int w = static_cast<int>(i % width);
    const long long th = i / width;
    const int h = static_cast<int>(th % heads);
    const int t = static_cast<int>(th / heads);
    out[i] = static_cast<float>(
        in[(static_cast<std::size_t>(h) * tokens + t) * width + w]);
  }
}

__global__ __launch_bounds__(kScoreWaves* kWaveLanes) void DsaScoresKernel(
    const float* __restrict__ iq, const float* __restrict__ proj,
    const float* __restrict__ pooled, int npool, float* __restrict__ scores,
    int pos0, long long score_stride, const int* __restrict__ pos_dev) {
  const int token = blockIdx.y;
  if (pos_dev != nullptr) {
    npool = (*pos_dev + 1) / kDsaPool;
  }
  if (blockIdx.x * kScoreWaves >= npool && pos0 < 0) {
    return;
  }
  if (pos0 >= 0) {
    npool = (pos0 + token + 1) / kDsaPool;
    iq += static_cast<std::size_t>(token) * kDsaIndexHeads * kDsaIndexDim;
    proj += static_cast<std::size_t>(token) * kDsaIndexHeads;
    scores += token * score_stride;
  }
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

constexpr int kScoreTokens = 16;
constexpr int kScoreWmmaWaves = 8;
constexpr int kScoreHeadGroup = 4;
constexpr int kScoreQPad = kDsaIndexDim + 8;

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__global__
__launch_bounds__(kScoreWmmaWaves* kWaveLanes) void DsaScoresWmmaKernel(
    const float* __restrict__ iq, const float* __restrict__ proj,
    const float* __restrict__ pooled, int pos0, int tokens, int max_pools,
    long long score_stride, float* __restrict__ scores) {
  const int tok0 = blockIdx.y * kScoreTokens;
  const int last_pos = pos0 + min(tok0 + kScoreTokens, tokens) - 1;
  if (blockIdx.x * kScoreWmmaWaves * 16 >= (last_pos + 1) / kDsaPool) {
    return;
  }
  const int tid = threadIdx.x;
  const int wave = tid / kWaveLanes;
  const int lane = tid % kWaveLanes;
  const int col = lane & 15;
  const int row_hi = lane >> 4;
  const int pool = (blockIdx.x * kScoreWmmaWaves + wave) * 16 + col;

  __shared__ _Float16 qh[kScoreTokens][kScoreHeadGroup][kScoreQPad];
  __shared__ _Float16 ql[kScoreTokens][kScoreHeadGroup][kScoreQPad];
  __shared__ float wt[kScoreTokens][kDsaIndexHeads];
  for (int idx = tid; idx < kScoreTokens * kDsaIndexHeads;
       idx += kScoreWmmaWaves * kWaveLanes) {
    const int t = idx / kDsaIndexHeads;
    const int h = idx % kDsaIndexHeads;
    const int tok = tok0 + t;
    wt[t][h] =
        tok < tokens
            ? proj[static_cast<std::size_t>(tok) * kDsaIndexHeads + h] *
                  rsqrtf(static_cast<float>(kDsaIndexDim * kDsaIndexHeads))
            : 0.0F;
  }

  v16h bh[kDsaIndexDim / 16];
  v16h bl[kDsaIndexDim / 16];
#pragma unroll
  for (int step = 0; step < kDsaIndexDim / 16; ++step) {
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      const float f =
          pool < max_pools
              ? pooled[static_cast<std::size_t>(pool) * kDsaIndexDim +
                       step * 16 + e]
              : 0.0F;
      const _Float16 hi = static_cast<_Float16>(f);
      bh[step][e] = hi;
      bl[step][e] = static_cast<_Float16>(f - static_cast<float>(hi));
    }
  }
  v8f total = v8f{};
  for (int group = 0; group < kDsaIndexHeads / kScoreHeadGroup; ++group) {
    __syncthreads();
    for (int idx = tid; idx < kScoreTokens * kScoreHeadGroup * kDsaIndexDim;
         idx += kScoreWmmaWaves * kWaveLanes) {
      const int d = idx % kDsaIndexDim;
      const int hh = (idx / kDsaIndexDim) % kScoreHeadGroup;
      const int t = idx / (kDsaIndexDim * kScoreHeadGroup);
      const int tok = tok0 + t;
      const float f = tok < tokens
                          ? iq[(static_cast<std::size_t>(tok) * kDsaIndexHeads +
                                group * kScoreHeadGroup + hh) *
                                   kDsaIndexDim +
                               d]
                          : 0.0F;
      const _Float16 hi = static_cast<_Float16>(f);
      qh[t][hh][d] = hi;
      ql[t][hh][d] = static_cast<_Float16>(f - static_cast<float>(hi));
    }
    __syncthreads();
    for (int hh = 0; hh < kScoreHeadGroup; ++hh) {
      v8f acc = v8f{};
#pragma unroll
      for (int step = 0; step < kDsaIndexDim / 16; ++step) {
        const v16h a = *reinterpret_cast<const v16h*>(&qh[col][hh][step * 16]);
        const v16h a_lo =
            *reinterpret_cast<const v16h*>(&ql[col][hh][step * 16]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bh[step], acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bl[step], acc);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, bh[step], acc);
      }
      for (int i = 0; i < 8; ++i) {
        total[i] += fmaxf(acc[i], 0.0F) *
                    wt[2 * i + row_hi][group * kScoreHeadGroup + hh];
      }
    }
  }
  for (int i = 0; i < 8; ++i) {
    const int tok = tok0 + 2 * i + row_hi;
    if (tok < tokens && pool < (pos0 + tok + 1) / kDsaPool) {
      scores[tok * score_stride + pool] = total[i];
    }
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
    int* __restrict__ selected, int pos0, long long score_stride,
    const int* __restrict__ pos_dev) {
  if (pos_dev != nullptr) {
    npool = (*pos_dev + 1) / kDsaPool;
  }
  if (pos0 >= 0) {
    const int token = blockIdx.x;
    npool = (pos0 + token + 1) / kDsaPool;
    scores += token * score_stride;
    selected += static_cast<std::size_t>(token) * kDsaTopPools;
  }
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
    float* __restrict__ ctx, int pos0, int tokens, int top,
    const int* __restrict__ pos_dev) {
  const int token = blockIdx.y;
  if (pos_dev != nullptr) {
    pos = *pos_dev;
    nselected = min((pos + 1) / kDsaPool, top);
  }
  if (pos0 >= 0) {
    pos = pos0 + token;
    nselected = min((pos + 1) / kDsaPool, top);
    selected += static_cast<std::size_t>(token) * kDsaTopPools;
  }
  const std::size_t slab =
      (static_cast<std::size_t>(blockIdx.x) * tokens + token) * kDsaLatent;
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
    q[c] = qabs[slab + c];
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
    ctx[slab + c] = acc;
  }
}

constexpr int kAttnKeys = 32;
constexpr int kAttnGather = kAttnKeys * (kDsaLatent / 8) / (8 * 32);
constexpr int kAttnWaves = 8;
constexpr int kAttnRowPad = kDsaLatent + 8;
constexpr int kAttnGroups = kDsaHeads / 16;
constexpr int kAttnWaveTiles = kDsaLatent / kAttnWaves / 16;
constexpr int kAttnChanTiles = kAttnGroups * kAttnWaveTiles;

// Absorbed-MLA attention with WMMA: each block takes one token and all 64
// heads; wave pairs own a 16-head group, the pair splitting the 512 latent
// channels. Selected latent rows are gathered once per chunk of 32 keys into
// LDS as F16 and serve as both keys and values; softmax is online over chunks.
__global__ __launch_bounds__(kAttnWaves* kWaveLanes) void DsaAttendWmmaKernel(
    const _Float16* __restrict__ q_hi, const _Float16* __restrict__ q_lo,
    const _Float16* __restrict__ lat, const int* __restrict__ selected,
    int pos0, int tokens, int top, float scale, _Float16* __restrict__ ctx) {
  const int token = blockIdx.x;
  const int pos = pos0 + token;
  const int nselected = min((pos + 1) / kDsaPool, top);
  selected += static_cast<std::size_t>(token) * kDsaTopPools;
  const int tail = (pos + 1) % kDsaPool;
  const int nvis = nselected * kDsaPool + tail;
  const int tid = threadIdx.x;
  const int wave = tid / kWaveLanes;
  const int lane = tid % kWaveLanes;
  const int group = wave >> 1;
  const int half = wave & 1;
  const int col = lane & 15;
  const int row_hi = lane >> 4;

  __shared__ _Float16 rows[kAttnKeys][kAttnRowPad];
  __shared__ float score[kAttnGroups][2][16][kAttnKeys + 1];
  __shared__ _Float16 prob[kAttnGroups][16][kAttnKeys];
  __shared__ float run_max[kAttnGroups * 16];
  __shared__ float alpha[kAttnGroups * 16];
  __shared__ float denom[kAttnGroups * 16];
  if (tid < kAttnGroups * 16) {
    run_max[tid] = -1e30F;
    denom[tid] = 0.0F;
  }
  v8f acc[kAttnChanTiles];
  for (int t = 0; t < kAttnChanTiles; ++t) {
    acc[t] = v8f{};
  }
  const std::size_t q_off =
      (static_cast<std::size_t>(group * 16 + col) * tokens + token) *
      kDsaLatent;

  uint4 prefetch[kAttnGather];
  const auto fetch = [&](int base) {
    for (int i = 0; i < kAttnGather; ++i) {
      const int j = (tid >> 6) + kAttnWaves * kWaveLanes / 64 * i;
      const int p = base + j;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (p < nvis) {
        const int row = p < nselected * kDsaPool
                            ? selected[p / kDsaPool] * kDsaPool + (p % kDsaPool)
                            : pos - (p - nselected * kDsaPool);
        v = *reinterpret_cast<const uint4*>(
            lat + static_cast<std::size_t>(row) * kDsaLatent + (tid & 63) * 8);
      }
      prefetch[i] = v;
    }
  };
  fetch(0);
  for (int base = 0; base < nvis; base += kAttnKeys) {
    const int n = min(kAttnKeys, nvis - base);
    __syncthreads();
    for (int i = 0; i < kAttnGather; ++i) {
      const int j = (tid >> 6) + kAttnWaves * kWaveLanes / 64 * i;
      *reinterpret_cast<uint4*>(&rows[j][(tid & 63) * 8]) = prefetch[i];
    }
    __syncthreads();
    {
      v8f s0 = v8f{};
      v8f s1 = v8f{};
      const _Float16* k0 = &rows[col][0];
      const _Float16* k1 = &rows[16 + col][0];
      for (int step = half * (kDsaLatent / 32);
           step < (half + 1) * (kDsaLatent / 32); ++step) {
        const v16h a = *reinterpret_cast<const v16h*>(q_hi + q_off + step * 16);
        const v16h a_lo =
            *reinterpret_cast<const v16h*>(q_lo + q_off + step * 16);
        const v16h b0 = *reinterpret_cast<const v16h*>(k0 + step * 16);
        const v16h b1 = *reinterpret_cast<const v16h*>(k1 + step * 16);
        s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b0, s0);
        s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b1, s1);
        s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b0, s0);
        s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo, b1, s1);
      }
      for (int i = 0; i < 8; ++i) {
        score[group][half][2 * i + row_hi][col] = s0[i] * scale;
        score[group][half][2 * i + row_hi][16 + col] = s1[i] * scale;
      }
    }
    __syncthreads();
    {
      const int gh = tid >> 2;
      const int part = tid & 3;
      const int g = gh >> 4;
      const int h = gh & 15;
      constexpr int kPerLane = kAttnKeys / 4;
      float best = -1e30F;
      for (int j = part * kPerLane; j < (part + 1) * kPerLane; ++j) {
        if (j < n) {
          best = fmaxf(best, score[g][0][h][j] + score[g][1][h][j]);
        }
      }
      best = fmaxf(best, XorLane(best, 1));
      best = fmaxf(best, XorLane(best, 2));
      const float old_max = run_max[gh];
      const float next = fmaxf(old_max, best);
      float sum = 0.0F;
      for (int j = part * kPerLane; j < (part + 1) * kPerLane; ++j) {
        const float p =
            j < n ? expf(score[g][0][h][j] + score[g][1][h][j] - next) : 0.0F;
        prob[g][h][j] = static_cast<_Float16>(p);
        sum += p;
      }
      sum += XorLane(sum, 1);
      sum += XorLane(sum, 2);
      if (part == 0) {
        const float scale_old = expf(old_max - next);
        alpha[gh] = scale_old;
        run_max[gh] = next;
        denom[gh] = denom[gh] * scale_old + sum;
      }
    }
    __syncthreads();
    fetch(base + kAttnKeys);
    for (int g = 0; g < kAttnGroups; ++g) {
      float a_scale[8];
      for (int i = 0; i < 8; ++i) {
        a_scale[i] = alpha[g * 16 + 2 * i + row_hi];
      }
      for (int t = 0; t < kAttnWaveTiles; ++t) {
        for (int i = 0; i < 8; ++i) {
          acc[g * kAttnWaveTiles + t][i] *= a_scale[i];
        }
      }
    }
    for (int step = 0; step < kAttnKeys / 16; ++step) {
      v16h b[kAttnWaveTiles];
      for (int t = 0; t < kAttnWaveTiles; ++t) {
        const int c = wave * (kAttnWaveTiles * 16) + t * 16 + col;
        for (int k = 0; k < 16; ++k) {
          b[t][k] = rows[step * 16 + k][c];
        }
      }
      for (int g = 0; g < kAttnGroups; ++g) {
        const v16h a = *reinterpret_cast<const v16h*>(&prob[g][col][step * 16]);
        for (int t = 0; t < kAttnWaveTiles; ++t) {
          acc[g * kAttnWaveTiles + t] =
              __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                  a, b[t], acc[g * kAttnWaveTiles + t]);
        }
      }
    }
  }
  __syncthreads();
  for (int g = 0; g < kAttnGroups; ++g) {
    for (int t = 0; t < kAttnWaveTiles; ++t) {
      const int c = wave * (kAttnWaveTiles * 16) + t * 16 + col;
      for (int i = 0; i < 8; ++i) {
        const int h = g * 16 + 2 * i + row_hi;
        const float inv = 1.0F / denom[h];
        ctx[(static_cast<std::size_t>(h) * tokens + token) * kDsaLatent + c] =
            static_cast<_Float16>(acc[g * kAttnWaveTiles + t][i] * inv);
      }
    }
  }
}

}  // namespace

int RmsNorm(float* x, const float* w, int n, float eps, hipStream_t stream,
            int rows) {
  RmsNormKernel<<<rows, kThreads, 0, stream>>>(x, w, n, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int LayerNorm(float* x, const float* w, const float* b, int n, float eps,
              hipStream_t stream, int rows) {
  LayerNormKernel<<<rows, kThreads, 0, stream>>>(x, w, b, n, eps);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAppend(const DsaCache& cache, const float* lat, const float* ik,
              const float* gate, const float* ape, int pos,
              hipStream_t stream) {
  DsaAppendKernel<<<1, kDsaIndexDim, 0, stream>>>(cache, lat, ik, gate, ape,
                                                  pos, nullptr);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaScores(const float* iq, const float* proj, const float* pooled,
              int npool, float* scores, hipStream_t stream) {
  if (npool <= 0) {
    return 0;
  }
  DsaScoresKernel<<<(npool + kScoreWaves - 1) / kScoreWaves,
                    kScoreWaves * kWaveLanes, 0, stream>>>(
      iq, proj, pooled, npool, scores, -1, 0, nullptr);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaSelect(const float* scores, int npool, int top_pools, int* selected,
              hipStream_t stream) {
  if (npool <= 0) {
    return 0;
  }
  DsaSelectKernel<<<1, kThreads, 0, stream>>>(scores, npool, top_pools,
                                              selected, -1, 0, nullptr);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAttend(const float* qabs, const float* lat, const int* selected,
              int nselected, int pos, float scale, float* ctx,
              hipStream_t stream) {
  DsaAttendKernel<<<kDsaHeads, kAttendThreads, 0, stream>>>(
      qabs, lat, selected, nselected, pos, scale, ctx, -1, 1, 0, nullptr);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAppendDev(const DsaCache& cache, const float* lat, const float* ik,
                 const float* gate, const float* ape, const int* pos,
                 hipStream_t stream) {
  DsaAppendKernel<<<1, kDsaIndexDim, 0, stream>>>(cache, lat, ik, gate, ape, 0,
                                                  pos);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaScoresDev(const float* iq, const float* proj, const float* pooled,
                 int max_pools, float* scores, const int* pos,
                 hipStream_t stream) {
  if (max_pools <= 0) {
    return 0;
  }
  DsaScoresKernel<<<(max_pools + kScoreWaves - 1) / kScoreWaves,
                    kScoreWaves * kWaveLanes, 0, stream>>>(iq, proj, pooled, 0,
                                                           scores, -1, 0, pos);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaSelectDev(const float* scores, int top_pools, int* selected,
                 const int* pos, hipStream_t stream) {
  DsaSelectKernel<<<1, kThreads, 0, stream>>>(scores, 0, top_pools, selected,
                                              -1, 0, pos);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAttendDev(const float* qabs, const float* lat, const int* selected,
                 int top_pools, float scale, float* ctx, const int* pos,
                 hipStream_t stream) {
  DsaAttendKernel<<<kDsaHeads, kAttendThreads, 0, stream>>>(
      qabs, lat, selected, 0, 0, scale, ctx, -1, 1, top_pools, pos);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAppendBatch(const DsaCache& cache, const float* lat, const float* ik,
                   const float* gate, const float* ape, int pos0, int tokens,
                   hipStream_t stream) {
  DsaAppendRowsKernel<<<tokens, kDsaIndexDim, 0, stream>>>(cache, lat, ik, gate,
                                                           pos0);
  const int first_pool = pos0 / kDsaPool;
  const int pools = (pos0 + tokens) / kDsaPool - first_pool;
  if (pools > 0) {
    DsaPoolKernel<<<pools, kDsaIndexDim, 0, stream>>>(cache, ape, first_pool);
  }
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaScoresBatch(const float* iq, const float* proj, const float* pooled,
                   int pos0, int tokens, long long score_stride, float* scores,
                   hipStream_t stream) {
  const int max_pools = (pos0 + tokens) / kDsaPool;
  if (max_pools <= 0) {
    return 0;
  }
  DsaScoresWmmaKernel<<<dim3((max_pools + kScoreWmmaWaves * 16 - 1) /
                                 (kScoreWmmaWaves * 16),
                             (tokens + kScoreTokens - 1) / kScoreTokens),
                        kScoreWmmaWaves * kWaveLanes, 0, stream>>>(
      iq, proj, pooled, pos0, tokens, max_pools, score_stride, scores);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaSelectBatch(const float* scores, long long score_stride, int pos0,
                   int tokens, int top_pools, int* selected,
                   hipStream_t stream) {
  if ((pos0 + tokens) / kDsaPool <= 0) {
    return 0;
  }
  DsaSelectKernel<<<tokens, kThreads, 0, stream>>>(
      scores, 0, top_pools, selected, pos0, score_stride, nullptr);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int DsaAttendBatch(const std::uint16_t* q_hi, const std::uint16_t* q_lo,
                   const std::uint16_t* lat_half, const int* selected, int pos0,
                   int tokens, int top_pools, float scale,
                   std::uint16_t* ctx_half, hipStream_t stream) {
  DsaAttendWmmaKernel<<<tokens, kAttnWaves * kWaveLanes, 0, stream>>>(
      reinterpret_cast<const _Float16*>(q_hi),
      reinterpret_cast<const _Float16*>(q_lo),
      reinterpret_cast<const _Float16*>(lat_half), selected, pos0, tokens,
      top_pools, scale, reinterpret_cast<_Float16*>(ctx_half));
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HeadMajor(const float* in, std::uint16_t* out, int tokens, int heads,
              int width, hipStream_t stream) {
  HeadMajorKernel<<<2048, 256, 0, stream>>>(
      in, reinterpret_cast<_Float16*>(out), tokens, heads, width);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int TokenMajor(const std::uint16_t* in, float* out, int tokens, int heads,
               int width, hipStream_t stream) {
  TokenMajorKernel<<<2048, 256, 0, stream>>>(
      reinterpret_cast<const _Float16*>(in), out, tokens, heads, width);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
