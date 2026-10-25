#include "src/models/flint/gpu/kernels/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gufo::models::flint::gpu::rocm {
namespace {

constexpr int kBlock = 256;
constexpr int kAttnBlock = 128;
constexpr int kQ8Block = 32;
constexpr int kQ8Bytes = 34;
constexpr float kNegInf = -3.0e38F;

unsigned Blocks(std::size_t count, int block = kBlock) {
  return static_cast<unsigned>((count + block - 1) / block);
}

__device__ float Q8Value(const unsigned char* row, std::uint32_t i) {
  const unsigned char* b = row + (i / kQ8Block) * kQ8Bytes;
  const float d = __half2float(*reinterpret_cast<const __half*>(b));
  return d * static_cast<float>(static_cast<signed char>(b[2 + i % kQ8Block]));
}

__global__ void EmbedQ8Kernel(const unsigned char* table,
                              const std::int32_t* tokens, float* out,
                              std::uint32_t hidden) {
  const unsigned char* row =
      table + static_cast<std::size_t>(tokens[blockIdx.x]) *
                  (hidden / kQ8Block) * kQ8Bytes;
  float* dst = out + static_cast<std::size_t>(blockIdx.x) * hidden;
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    dst[i] = Q8Value(row, i);
  }
}

__device__ float BlockSum(float v, float* scratch) {
  scratch[threadIdx.x] = v;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      scratch[threadIdx.x] += scratch[threadIdx.x + s];
    __syncthreads();
  }
  const float total = scratch[0];
  __syncthreads();
  return total;
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t dim, std::uint32_t groups,
                              float eps) {
  __shared__ float scratch[kBlock];
  const std::uint32_t width = dim / groups;
  const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim +
                           static_cast<std::size_t>(blockIdx.y) * width;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    sum += x[base + i] * x[base + i];
  }
  const float total = BlockSum(sum, scratch);
  const float inv = 1.0F / sqrtf(total / static_cast<float>(width) + eps);
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    out[base + i] = x[base + i] * inv * (gamma != nullptr ? gamma[i] : 1.0F);
  }
}

__device__ float WaveSum(float v) {
  for (int off = warpSize / 2; off > 0; off >>= 1)
    v += __shfl_xor(v, off);
  return v;
}

__device__ float BlockSumWaves(float v, float* scratch) {
  const unsigned lane = threadIdx.x % warpSize;
  const unsigned wave = threadIdx.x / warpSize;
  v = WaveSum(v);
  if (lane == 0)
    scratch[wave] = v;
  __syncthreads();
  float total = 0.0F;
  for (unsigned w = 0; w < blockDim.x / warpSize; ++w)
    total += scratch[w];
  __syncthreads();
  return total;
}

// Writes one Q8_1 block's int8 values and (d, sum) header. `v` holds the 8
// values a lane owns; four consecutive lanes cover one 32-value block.
__device__ void StoreQ8x8(unsigned char* row, std::uint32_t base,
                          const float (&v)[8]) {
  float amax = 0.0F;
  float sum = 0.0F;
  for (int j = 0; j < 8; ++j) {
    amax = fmaxf(amax, fabsf(v[j]));
    sum += v[j];
  }
  for (int off = 1; off < 4; off <<= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off));
    sum += __shfl_xor(sum, off);
  }
  const float d = amax / 127.0F;
  unsigned packed[2] = {0, 0};
  for (int j = 0; j < 8; ++j) {
    const int q = amax == 0.0F ? 0 : static_cast<int>(roundf(v[j] / d));
    packed[j / 4] |= (static_cast<unsigned>(q) & 0xffU) << (8 * (j % 4));
  }
  unsigned char* block = row + (base / 32) * 36;
  unsigned* qs = reinterpret_cast<unsigned*>(block + 4 + base % 32);
  qs[0] = packed[0];
  qs[1] = packed[1];
  if (threadIdx.x % 4 == 0) {
    *reinterpret_cast<__half2*>(block) =
        __floats2half2_rn(d, fminf(fmaxf(sum, -65504.0F), 65504.0F));
  }
}

__global__ void ResidualNormQuantKernel(float* x, const float* delta,
                                        const float* gamma, float* xn,
                                        unsigned char* q8, std::uint32_t dim,
                                        float eps) {
  __shared__ float scratch[kBlock / 32];
  const std::size_t row = blockIdx.x;
  const std::uint32_t e = threadIdx.x * 8;
  float v[8];
  float sq = 0.0F;
  if (e < dim) {
    const float4 a = *reinterpret_cast<const float4*>(x + row * dim + e);
    const float4 b = *reinterpret_cast<const float4*>(x + row * dim + e + 4);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
    if (delta != nullptr) {
      const float4 c = *reinterpret_cast<const float4*>(delta + row * dim + e);
      const float4 f =
          *reinterpret_cast<const float4*>(delta + row * dim + e + 4);
      v[0] += c.x;
      v[1] += c.y;
      v[2] += c.z;
      v[3] += c.w;
      v[4] += f.x;
      v[5] += f.y;
      v[6] += f.z;
      v[7] += f.w;
      *reinterpret_cast<float4*>(x + row * dim + e) =
          make_float4(v[0], v[1], v[2], v[3]);
      *reinterpret_cast<float4*>(x + row * dim + e + 4) =
          make_float4(v[4], v[5], v[6], v[7]);
    }
    for (int j = 0; j < 8; ++j)
      sq += v[j] * v[j];
  } else {
    for (int j = 0; j < 8; ++j)
      v[j] = 0.0F;
  }
  const float total = BlockSumWaves(sq, scratch);
  const float inv = 1.0F / sqrtf(total / static_cast<float>(dim) + eps);
  if (e < dim) {
    float y[8];
    for (int j = 0; j < 8; ++j)
      y[j] = v[j] * inv * gamma[e + j];
    *reinterpret_cast<float4*>(xn + row * dim + e) =
        make_float4(y[0], y[1], y[2], y[3]);
    *reinterpret_cast<float4*>(xn + row * dim + e + 4) =
        make_float4(y[4], y[5], y[6], y[7]);
    if (q8 != nullptr)
      StoreQ8x8(q8 + row * (dim / 32) * 36, e, y);
  }
}

__global__ void MoeCombineNormQuantKernel(
    float* x, const float* expert_out, const float* weights,
    const float* expert_scale, const float* shared, const float* shared_scale,
    std::uint32_t k, const float* gamma, float* xn, unsigned char* q8,
    std::uint32_t dim, float eps) {
  __shared__ float scratch[kBlock / 32];
  const std::size_t row = blockIdx.x;
  const std::uint32_t e = threadIdx.x * 8;
  float v[8];
  float sq = 0.0F;
  if (e < dim) {
    const float4 a = *reinterpret_cast<const float4*>(x + row * dim + e);
    const float4 b = *reinterpret_cast<const float4*>(x + row * dim + e + 4);
    const float4 sa = *reinterpret_cast<const float4*>(shared + row * dim + e);
    const float4 sb =
        *reinterpret_cast<const float4*>(shared + row * dim + e + 4);
    const float ss = shared_scale != nullptr ? shared_scale[row] : 1.0F;
    float acc[8] = {sa.x * ss, sa.y * ss, sa.z * ss, sa.w * ss,
                    sb.x * ss, sb.y * ss, sb.z * ss, sb.w * ss};
    for (std::uint32_t s = 0; s < k; ++s) {
      const float w =
          weights[row * k + s] *
          (expert_scale != nullptr ? expert_scale[row * k + s] : 1.0F);
      const float* src = expert_out + (row * k + s) * dim + e;
      const float4 ea = *reinterpret_cast<const float4*>(src);
      const float4 eb = *reinterpret_cast<const float4*>(src + 4);
      acc[0] += w * ea.x;
      acc[1] += w * ea.y;
      acc[2] += w * ea.z;
      acc[3] += w * ea.w;
      acc[4] += w * eb.x;
      acc[5] += w * eb.y;
      acc[6] += w * eb.z;
      acc[7] += w * eb.w;
    }
    v[0] = a.x + acc[0];
    v[1] = a.y + acc[1];
    v[2] = a.z + acc[2];
    v[3] = a.w + acc[3];
    v[4] = b.x + acc[4];
    v[5] = b.y + acc[5];
    v[6] = b.z + acc[6];
    v[7] = b.w + acc[7];
    *reinterpret_cast<float4*>(x + row * dim + e) =
        make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(x + row * dim + e + 4) =
        make_float4(v[4], v[5], v[6], v[7]);
    for (int j = 0; j < 8; ++j)
      sq += v[j] * v[j];
  } else {
    for (int j = 0; j < 8; ++j)
      v[j] = 0.0F;
  }
  const float total = BlockSumWaves(sq, scratch);
  const float inv = 1.0F / sqrtf(total / static_cast<float>(dim) + eps);
  if (e < dim) {
    float y[8];
    for (int j = 0; j < 8; ++j)
      y[j] = v[j] * inv * gamma[e + j];
    *reinterpret_cast<float4*>(xn + row * dim + e) =
        make_float4(y[0], y[1], y[2], y[3]);
    *reinterpret_cast<float4*>(xn + row * dim + e + 4) =
        make_float4(y[4], y[5], y[6], y[7]);
    if (q8 != nullptr)
      StoreQ8x8(q8 + row * (dim / 32) * 36, e, y);
  }
}

// One block per (head, token); head < heads are query heads, then kv_heads key
// heads, then kv_heads value heads. Thread d owns dimension d.
__global__ void QkvPrepareKernel(QkvSources src, const float* q_gamma,
                                 const float* k_gamma, float* q_out,
                                 __half* k_cache, __half* v_cache,
                                 std::uint32_t heads, std::uint32_t kv_heads,
                                 std::uint32_t d,
                                 const std::uint32_t* start_pos, float base,
                                 float eps, bool rope, __half* q_half) {
  __shared__ float scratch[kAttnBlock / 32];
  __shared__ float partner[kAttnBlock];
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t pos = *start_pos + t;
  const bool is_q = h < heads;
  const bool is_k = !is_q && h < heads + kv_heads;
  float val;
  if (is_q) {
    val = src.q[static_cast<std::size_t>(t) * src.q_stride + h * d + i];
  } else if (is_k) {
    val =
        src.k[static_cast<std::size_t>(t) * src.k_stride + (h - heads) * d + i];
  } else {
    val = src.v[static_cast<std::size_t>(t) * src.v_stride +
                (h - heads - kv_heads) * d + i];
  }
  if (is_q || is_k) {
    const float total = BlockSumWaves(val * val, scratch);
    val = val * (1.0F / sqrtf(total / static_cast<float>(d) + eps)) *
          (is_q ? q_gamma[i] : k_gamma[i]);
    if (rope) {
      partner[i] = val;
      __syncthreads();
      const std::uint32_t half = d / 2;
      const std::uint32_t p = i % half;
      const float theta = static_cast<float>(pos) *
                          (1.0F / powf(base, static_cast<float>(2 * p) /
                                                 static_cast<float>(d)));
      float s = 0.0F;
      float c = 0.0F;
      sincosf(theta, &s, &c);
      const float x0 = partner[p];
      const float x1 = partner[p + half];
      val = i < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
  }
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * d;
  if (is_q) {
    if (q_half != nullptr) {
      q_half[(static_cast<std::size_t>(t) * heads + h) * d + i] =
          __float2half(val);
    } else {
      q_out[(static_cast<std::size_t>(t) * heads + h) * d + i] = val;
    }
  } else if (is_k) {
    k_cache[pos * kv_row + (h - heads) * d + i] = __float2half(val);
  } else {
    v_cache[pos * kv_row + (h - heads - kv_heads) * d + i] = __float2half(val);
  }
}

__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           const std::uint32_t* start_pos, float base) {
  const std::uint32_t pairs = d / 2;
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * pairs)
    return;
  const std::uint32_t head = idx / pairs;
  const std::uint32_t pair = idx % pairs;
  const std::uint32_t i = 2 * pair;
  const float pos = static_cast<float>(*start_pos + blockIdx.y);
  const float theta =
      pos * (1.0F / powf(base, static_cast<float>(i) / static_cast<float>(d)));
  float s = 0.0F;
  float c = 0.0F;
  sincosf(theta, &s, &c);
  float* p = x + (static_cast<std::size_t>(blockIdx.y) * heads + head) * d;
  const float x0 = p[pair];
  const float x1 = p[pair + pairs];
  p[pair] = x0 * c - x1 * s;
  p[pair + pairs] = x0 * s + x1 * c;
}

__global__ void StoreKvKernel(const float* k, const float* v, __half* k_cache,
                              __half* v_cache, std::uint32_t row_dim,
                              const std::uint32_t* start_pos) {
  const std::size_t dst =
      static_cast<std::size_t>(*start_pos + blockIdx.y) * row_dim;
  const std::size_t src = static_cast<std::size_t>(blockIdx.y) * row_dim;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < row_dim;
       i += gridDim.x * blockDim.x) {
    k_cache[dst + i] = __float2half(k[src + i]);
    v_cache[dst + i] = __float2half(v[src + i]);
  }
}

// One block per (query head, token); thread d owns output dimension d.
__global__ void AttentionKernel(const float* q, const __half* k_cache,
                                const __half* v_cache, const float* sinks,
                                float* out, const std::uint32_t* start_pos,
                                std::uint32_t heads, std::uint32_t kv_heads,
                                std::uint32_t d, std::uint32_t window) {
  __shared__ float q_s[kAttnBlock];
  __shared__ float p_s[kAttnBlock];
  __shared__ float red[kAttnBlock];
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t pos = *start_pos + blockIdx.y;
  const std::uint32_t kv_head = head * kv_heads / heads;
  const std::size_t row = static_cast<std::size_t>(kv_heads) * d;
  const float scale = rsqrtf(static_cast<float>(d));
  const std::uint32_t first =
      (window > 0 && pos >= window) ? pos - window + 1 : 0;

  q_s[tid] = q[(static_cast<std::size_t>(blockIdx.y) * heads + head) * d + tid];
  float m = sinks[head];
  float l = 1.0F;
  float acc = 0.0F;
  __syncthreads();

  for (std::uint32_t tile = first; tile <= pos; tile += kAttnBlock) {
    const std::uint32_t key = tile + tid;
    float s = kNegInf;
    if (key <= pos) {
      const __half* kr = k_cache + key * row + kv_head * d;
      float dot = 0.0F;
      for (std::uint32_t i = 0; i < d; ++i) {
        dot += q_s[i] * __half2float(kr[i]);
      }
      s = dot * scale;
    }
    red[tid] = s;
    __syncthreads();
    for (unsigned stride = kAttnBlock / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        red[tid] = fmaxf(red[tid], red[tid + stride]);
      __syncthreads();
    }
    const float m_new = fmaxf(m, red[0]);
    __syncthreads();
    const float p = key <= pos ? expf(s - m_new) : 0.0F;
    p_s[tid] = p;
    red[tid] = p;
    __syncthreads();
    for (unsigned stride = kAttnBlock / 2; stride > 0; stride >>= 1) {
      if (tid < stride)
        red[tid] += red[tid + stride];
      __syncthreads();
    }
    const float alpha = expf(m - m_new);
    l = l * alpha + red[0];
    acc *= alpha;
    const std::uint32_t count =
        min(static_cast<std::uint32_t>(kAttnBlock), pos + 1 - tile);
    for (std::uint32_t j = 0; j < count; ++j) {
      acc +=
          p_s[j] * __half2float(v_cache[(tile + j) * row + kv_head * d + tid]);
    }
    m = m_new;
    __syncthreads();
  }
  out[(static_cast<std::size_t>(blockIdx.y) * heads + head) * d + tid] =
      acc / l;
}

// One block per (kv head, key slice, token). Each wave owns one query head of
// the GQA group; the block stages K/V tiles in LDS once for all of them. In the
// score pass a lane owns a key (full dot product from LDS), in the PV pass a
// lane owns four output dimensions.
constexpr int kGqaTile = 64;
constexpr int kGqaKPad = 130;  // halves per K row in LDS: odd word stride

__global__ void AttentionSliceKernel(const float* q, const __half* k_cache,
                                     const __half* v_cache, float* partial,
                                     const std::uint32_t* start_pos,
                                     std::uint32_t heads,
                                     std::uint32_t kv_heads, std::uint32_t d,
                                     std::uint32_t window,
                                     std::uint32_t splits) {
  __shared__ __half k_s[kGqaTile * kGqaKPad];
  __shared__ __half v_s[kGqaTile * 128];
  __shared__ float q_s[8 * 128];
  const std::uint32_t group = heads / kv_heads;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t split = blockIdx.y;
  const std::uint32_t lane = threadIdx.x % 32;
  const std::uint32_t wave = threadIdx.x / 32;
  const std::uint32_t head = kv_head * group + wave;
  const std::uint32_t pos = *start_pos + blockIdx.z;
  const std::size_t row = static_cast<std::size_t>(kv_heads) * d;
  const float scale = rsqrtf(static_cast<float>(d));
  const std::uint32_t first =
      (window > 0 && pos >= window) ? pos - window + 1 : 0;
  const std::uint32_t chunk = (pos + 1 - first + splits - 1) / splits;
  const std::uint32_t k0 = first + split * chunk;
  const std::uint32_t k1 = min(pos + 1, k0 + chunk);

  for (std::uint32_t i = threadIdx.x; i < group * d; i += blockDim.x) {
    q_s[i] =
        q[(static_cast<std::size_t>(blockIdx.z) * heads + kv_head * group) * d +
          i];
  }
  float m = kNegInf;
  float l = 0.0F;
  float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  for (std::uint32_t tile = k0; tile < k1; tile += kGqaTile) {
    const std::uint32_t cnt =
        min(static_cast<std::uint32_t>(kGqaTile), k1 - tile);
    __syncthreads();
    for (std::uint32_t i = threadIdx.x; i < cnt * (d / 8); i += blockDim.x) {
      const std::uint32_t key = i / (d / 8);
      const std::uint32_t c = i % (d / 8);
      const std::size_t src = (tile + key) * row + kv_head * d + c * 8;
      const uint4 kv = *reinterpret_cast<const uint4*>(k_cache + src);
      const uint4 vv = *reinterpret_cast<const uint4*>(v_cache + src);
      unsigned* kd = reinterpret_cast<unsigned*>(k_s + key * kGqaKPad + c * 8);
      kd[0] = kv.x;
      kd[1] = kv.y;
      kd[2] = kv.z;
      kd[3] = kv.w;
      *reinterpret_cast<uint4*>(v_s + key * 128 + c * 8) = vv;
    }
    __syncthreads();
    for (std::uint32_t sub = 0; sub < cnt; sub += 32) {
      const std::uint32_t key = sub + lane;
      const bool live = key < cnt;
      float s = kNegInf;
      if (live) {
        const unsigned* kr =
            reinterpret_cast<const unsigned*>(k_s + key * kGqaKPad);
        const float* qr = q_s + wave * 128;
        float dot = 0.0F;
        for (std::uint32_t i = 0; i < d / 2; ++i) {
          const unsigned w = kr[i];
          dot += qr[2 * i] * __half2float(__ushort_as_half(w & 0xffffU));
          dot += qr[2 * i + 1] * __half2float(__ushort_as_half(w >> 16));
        }
        s = dot * scale;
      }
      float tile_max = s;
      for (int off = 16; off > 0; off >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 32));
      }
      const float m_new = fmaxf(m, tile_max);
      const float p = live ? expf(s - m_new) : 0.0F;
      float psum = p;
      for (int off = 16; off > 0; off >>= 1)
        psum += __shfl_xor(psum, off, 32);
      const float alpha = expf(m - m_new);
      l = l * alpha + psum;
      for (int j = 0; j < 4; ++j)
        acc[j] *= alpha;
      const std::uint32_t live_keys = min(32U, cnt - sub);
      for (std::uint32_t j = 0; j < live_keys; ++j) {
        const float pj = __shfl(p, j, 32);
        const __half* vr = v_s + (sub + j) * 128 + lane * 4;
        acc[0] += pj * __half2float(vr[0]);
        acc[1] += pj * __half2float(vr[1]);
        acc[2] += pj * __half2float(vr[2]);
        acc[3] += pj * __half2float(vr[3]);
      }
      m = m_new;
    }
  }
  float* dst =
      partial +
      ((static_cast<std::size_t>(blockIdx.z) * heads + head) * splits + split) *
          (d + 2);
  for (int j = 0; j < 4; ++j)
    dst[lane * 4 + j] = acc[j];
  if (lane == 0) {
    dst[d] = m;
    dst[d + 1] = l;
  }
}

__global__ void AttentionMergeKernel(const float* partial, const float* sinks,
                                     float* out, unsigned char* out_q8,
                                     std::uint32_t heads, std::uint32_t d,
                                     std::uint32_t splits) {
  const std::uint32_t head = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const float* src =
      partial +
      (static_cast<std::size_t>(blockIdx.y) * heads + head) * splits * (d + 2);
  const float sink = sinks[head];
  float big = sink;
  for (std::uint32_t s = 0; s < splits; ++s) {
    big = fmaxf(big, src[s * (d + 2) + d]);
  }
  float denom = expf(sink - big);
  float acc = 0.0F;
  for (std::uint32_t s = 0; s < splits; ++s) {
    const float w = expf(src[s * (d + 2) + d] - big);
    denom += src[s * (d + 2) + d + 1] * w;
    acc += src[s * (d + 2) + tid] * w;
  }
  const float val = acc / denom;
  const std::size_t index =
      (static_cast<std::size_t>(blockIdx.y) * heads + head) * d + tid;
  if (out_q8 == nullptr) {
    out[index] = val;
    return;
  }
  // One Q8_1 block per 32 consecutive threads.
  float amax = fabsf(val);
  float sum = val;
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off, 32));
    sum += __shfl_xor(sum, off, 32);
  }
  const float scale = amax / 127.0F;
  const int q = amax == 0.0F ? 0 : static_cast<int>(roundf(val / scale));
  unsigned char* block = out_q8 + (index / 32) * 36;
  reinterpret_cast<signed char*>(block + 4)[index % 32] =
      static_cast<signed char>(q);
  if (index % 32 == 0) {
    *reinterpret_cast<__half2*>(block) =
        __floats2half2_rn(scale, fminf(fmaxf(sum, -65504.0F), 65504.0F));
  }
}

// Picks the k largest probabilities (lowest index wins ties), optionally
// renormalizes them and scales. Thread e holds expert e's probability, so
// n_experts <= blockDim.x. Each round reduces within waves by shuffles and
// across the waves through double-buffered shared memory (one sync a round).
__device__ void SelectTopK(float p, std::uint32_t n_experts, std::uint32_t k,
                           std::int32_t* ids, float* weights, bool normalize,
                           float scale) {
  constexpr int kMaxWaves = kBlock / 32;
  __shared__ float wave_v[2][kMaxWaves];
  __shared__ int wave_i[2][kMaxWaves];
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  const std::uint32_t waves = blockDim.x / warpSize;
  float v = threadIdx.x < n_experts ? p : kNegInf;
  float picked[16];
  for (std::uint32_t r = 0; r < k; ++r) {
    float best = v;
    int idx = static_cast<int>(threadIdx.x);
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(best, off);
      const int oi = __shfl_xor(idx, off);
      if (ov > best || (ov == best && oi < idx)) {
        best = ov;
        idx = oi;
      }
    }
    if (lane == 0) {
      wave_v[r & 1][wave] = best;
      wave_i[r & 1][wave] = idx;
    }
    __syncthreads();
    float win_v = wave_v[r & 1][0];
    int win_i = wave_i[r & 1][0];
    for (std::uint32_t w = 1; w < waves; ++w) {
      const float ov = wave_v[r & 1][w];
      const int oi = wave_i[r & 1][w];
      if (ov > win_v || (ov == win_v && oi < win_i)) {
        win_v = ov;
        win_i = oi;
      }
    }
    if (static_cast<int>(threadIdx.x) == win_i)
      v = kNegInf;
    picked[r] = win_v;
    if (threadIdx.x == 0)
      ids[r] = win_i;
  }
  if (threadIdx.x == 0) {
    float sum = 0.0F;
    for (std::uint32_t r = 0; r < k; ++r)
      sum += picked[r];
    const float denom = normalize ? fmaxf(sum, 6.103515625e-5F) : 1.0F;
    for (std::uint32_t r = 0; r < k; ++r) {
      weights[r] = picked[r] / denom * scale;
    }
  }
}

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

// Sixteen halves from 16-byte aligned memory.
__device__ __forceinline__ v16h LoadFrag(const __half* p) {
  const uint4 lo = *reinterpret_cast<const uint4*>(p);
  const uint4 hi = *(reinterpret_cast<const uint4*>(p) + 1);
  v16h r;
  __builtin_memcpy(&r, &lo, 16);
  __builtin_memcpy(reinterpret_cast<char*>(&r) + 16, &hi, 16);
  return r;
}

constexpr int kWmmaKeys = 32;
constexpr int kNb = kWmmaKeys / 16;
constexpr int kWmmaHeadDim = 128;
constexpr int kWmmaGroup = 8;
constexpr int kWmmaWaves = 8;
constexpr int kWmmaThreads = kWmmaWaves * 32;
constexpr int kWmmaTokens = 2 * kWmmaWaves;   // tokens per block: 2 per wave
constexpr int kWmmaKStride = 136;             // halves per K row in LDS
constexpr int kWmmaVtStride = kWmmaKeys + 8;  // halves per V^T row in LDS
constexpr int kWmmaPStride = kWmmaKeys + 8;

// One block per (kv head, group of 16 tokens). Wave w owns the queries of
// tokens t0 + 2w and t0 + 2w + 1 across the eight query heads (query =
// token * 8 + head, one per lane pair). The scores are computed transposed,
// S^T = K Q^T, so a lane holds one query and eight keys per 16-key block:
// the softmax max and sum are in-register, the probabilities are already the
// B operand of O^T = V^T P^T (after one lane swap), and no row statistics ever
// cross lanes except one max per tile.
constexpr std::uint32_t kSparseBlock = 64;
constexpr std::uint32_t kSparseSink = 2;
constexpr std::uint32_t kSparseLocal = 32;
constexpr std::uint32_t kSparseList = 512;

// Sixteen halves from 16-byte aligned global memory.
__device__ __forceinline__ v16h LoadFragGlobal(const __half* p) {
  const uint4 lo = *reinterpret_cast<const uint4*>(p);
  const uint4 hi = *(reinterpret_cast<const uint4*>(p) + 1);
  v16h r;
  __builtin_memcpy(&r, &lo, 16);
  __builtin_memcpy(reinterpret_cast<char*>(&r) + 16, &hi, 16);
  return r;
}

// Per 64-key block and kv head: the elementwise maximum and minimum of the
// keys (kmax then kmin, 128 halves each), the Quest bound summary. Blocks
// touched by the positions [start, start + n) are recomputed from the cache.
__global__ void PoolKeyBlocksKernel(const __half* __restrict__ k_cache,
                                    __half* __restrict__ kmm,
                                    const std::uint32_t* __restrict__ start_pos,
                                    std::uint32_t n_tokens,
                                    std::uint32_t kv_heads,
                                    std::uint32_t nblk) {
  const std::uint32_t first = *start_pos / kSparseBlock;
  const std::uint32_t block = first + blockIdx.x;
  const std::uint32_t last_key = *start_pos + n_tokens - 1;
  const std::uint32_t head = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * kWmmaHeadDim;
  const std::uint32_t key0 = block * kSparseBlock;
  const std::uint32_t key1 = min(key0 + kSparseBlock - 1, last_key);
  float hi = -65504.0F;
  float lo = 65504.0F;
  for (std::uint32_t key = key0; key <= key1; ++key) {
    const float v =
        __half2float(k_cache[key * kv_row + head * kWmmaHeadDim + d]);
    hi = fmaxf(hi, v);
    lo = fminf(lo, v);
  }
  __half* dst = kmm + (std::size_t{head} * nblk + block) * 256;
  dst[d] = __float2half(hi);
  dst[128 + d] = __float2half(lo);
}

template<bool kSparse>
__global__ __launch_bounds__(kWmmaThreads) void AttentionWmmaKernel(
    const __half* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const float* __restrict__ sinks,
    float* __restrict__ out, std::uint32_t n_tokens,
    const std::uint32_t* __restrict__ start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window,
    const __half* __restrict__ kmm, std::uint32_t nblk,
    std::uint32_t top_blocks) {
  // K tile and V^T tile; the block scores of the sparse prologue reuse them.
  constexpr int kKBytes = kWmmaKeys * kWmmaKStride * 2;
  constexpr int kVBytes = kWmmaHeadDim * kWmmaVtStride * 2;
  __shared__ __align__(16) char smem[kKBytes + kVBytes];
  auto* k_s = reinterpret_cast<__half*>(smem);
  auto* v_t = reinterpret_cast<__half*>(smem + kKBytes);
  __shared__ std::uint32_t blist[kSparseList];
  __shared__ std::uint32_t scan_tmp[kWmmaWaves + 2];
  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t wave = tid >> 5U;
  const std::uint32_t sub = lane & 15U;
  const std::uint32_t half_id = lane >> 4U;
  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t t0 = blockIdx.y * kWmmaTokens;
  const std::uint32_t start = *start_pos;
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * kWmmaHeadDim;
  constexpr float kLog2e = 1.4426950408889634F;
  // Scores and the running maximum are kept in log2 units.
  const float scale = 0.08838834764831845F * kLog2e;  // log2(e) / sqrt(128)

  // This lane's query: column `sub` of the 16 queries of the wave.
  const std::uint32_t token = t0 + 2 * wave + sub / kWmmaGroup;
  const std::uint32_t head = kv_head * kWmmaGroup + sub % kWmmaGroup;
  const bool live = token < n_tokens;
  const std::uint32_t qpos = start + token;

  v16h q_frag[8];
  {
    const __half* qr =
        q + (static_cast<std::size_t>(live ? token : 0) * heads + head) *
                kWmmaHeadDim;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      uint4 lo = make_uint4(0, 0, 0, 0);
      uint4 hi = lo;
      if (live) {
        lo = *reinterpret_cast<const uint4*>(qr + ks * 16);
        hi = *reinterpret_cast<const uint4*>(qr + ks * 16 + 8);
      }
      __builtin_memcpy(&q_frag[ks], &lo, 16);
      __builtin_memcpy(reinterpret_cast<char*>(&q_frag[ks]) + 16, &hi, 16);
    }
  }

  // v_perm selectors: low half-word from this lane's keys and high from the
  // partner's (lower lane half), or the reverse (upper lane half), for the
  // low (X = 0) and high (X = 1) half-word of a packed pair.
  std::uint32_t perm_sel[2];
#pragma unroll
  for (int x = 0; x < 2; ++x) {
    const std::uint32_t own_pair = (2 * x) | ((2 * x + 1) << 8);
    const std::uint32_t partner_pair = (4 + 2 * x) | ((5 + 2 * x) << 8);
    perm_sel[x] = half_id == 0 ? (own_pair | (partner_pair << 16))
                               : (partner_pair | (own_pair << 16));
  }

  float run_max = sinks[head] * kLog2e;
  // The sink counts once; the two lane halves each sum their own keys.
  float run_sum = half_id == 0 ? 1.0F : 0.0F;
  v8f o_acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t)
    o_acc[t] = v8f{};

  const std::uint32_t last_token = min(t0 + kWmmaTokens, n_tokens) - 1;
  const std::uint32_t pos_max = start + last_token;
  const std::uint32_t pos_min = start + t0;
  const std::uint32_t first_key =
      (window > 0 && pos_min >= window) ? pos_min - window + 1 : 0;

  // Key tiles to visit: a contiguous run, or (sparse) the sink blocks, the
  // best-scoring older blocks and the local window, each 64 keys.
  std::uint32_t n_tiles = (pos_max - first_key) / kWmmaKeys + 1;
  std::uint32_t list_count = 0;
  if constexpr (kSparse) {
    const std::uint32_t local_last = pos_max / kSparseBlock;
    const std::uint32_t pmin_block = pos_min / kSparseBlock;
    const std::uint32_t local_first =
        pmin_block >= kSparseLocal ? pmin_block - kSparseLocal : 0;
    const std::uint32_t cand_first = kSparseSink;
    const std::uint32_t cand_end = local_first;
    const std::uint32_t cand_count =
        cand_end > cand_first ? cand_end - cand_first : 0;
    auto* scores = reinterpret_cast<std::uint32_t*>(smem);
    if (cand_count <= top_blocks) {
      for (std::uint32_t b = tid; b <= local_last; b += kWmmaThreads)
        blist[b] = b;
      list_count = local_last + 1;
    } else {
      for (std::uint32_t b = tid; b < cand_count; b += kWmmaThreads)
        scores[b] = 0;
      __syncthreads();
      // Quest bound per block and query: q+ . kmax + q- . kmin, as two WMMA
      // chains with the pooled rows as the A operand.
      const v16h zero16 = v16h{};
      for (std::uint32_t t = wave; t * 16 < cand_count; t += kWmmaWaves) {
        std::uint32_t row = cand_first + t * 16 + sub;
        if (row >= cand_end)
          row = cand_end - 1;
        const __half* base = kmm + (std::size_t{kv_head} * nblk + row) * 256;
        v8f acc = v8f{};
#pragma unroll
        for (int ks = 0; ks < 8; ++ks) {
          const v16h a_max = LoadFragGlobal(base + ks * 16);
          const v16h a_min = LoadFragGlobal(base + 128 + ks * 16);
          acc = Wmma(a_max, __builtin_elementwise_max(q_frag[ks], zero16), acc);
          acc = Wmma(a_min, __builtin_elementwise_min(q_frag[ks], zero16), acc);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const std::uint32_t blk = cand_first + t * 16 + 2 * i + half_id;
          if (live && blk < cand_end) {
            const std::uint32_t bits = __float_as_uint(acc[i]);
            const std::uint32_t key =
                bits ^ ((bits >> 31) != 0 ? 0xFFFFFFFFU : 0x80000000U);
            atomicMax(scores + (blk - cand_first), key);
          }
        }
      }
      __syncthreads();
      // Largest threshold with at least top_blocks scores at or above it.
      std::uint32_t thr = 0;
      for (int bit = 31; bit >= 0; --bit) {
        const std::uint32_t trial = thr | (1U << bit);
        std::uint32_t cnt = 0;
        for (std::uint32_t b = tid; b < cand_count; b += kWmmaThreads)
          cnt += scores[b] >= trial ? 1U : 0U;
        for (int off = 16; off > 0; off >>= 1)
          cnt += __shfl_xor(cnt, off);
        if (lane == 0)
          scan_tmp[wave] = cnt;
        __syncthreads();
        std::uint32_t total = 0;
        for (int w = 0; w < kWmmaWaves; ++w)
          total += scan_tmp[w];
        __syncthreads();
        if (total >= top_blocks)
          thr = trial;
      }
      // Ordered compaction: scores above the threshold, then ties by index.
      const std::uint32_t per = (cand_count + kWmmaThreads - 1) / kWmmaThreads;
      const std::uint32_t b0 = tid * per;
      const std::uint32_t b1 = min(b0 + per, cand_count);
      std::uint32_t gt = 0;
      std::uint32_t eq = 0;
      for (std::uint32_t b = b0; b < b1; ++b) {
        gt += scores[b] > thr ? 1U : 0U;
        eq += scores[b] == thr ? 1U : 0U;
      }
      auto block_scan = [&](std::uint32_t value, std::uint32_t* total) {
        std::uint32_t inc = value;
        for (int off = 1; off < 32; off <<= 1) {
          const std::uint32_t up = __shfl_up(inc, off);
          if (lane >= static_cast<std::uint32_t>(off))
            inc += up;
        }
        if (lane == 31)
          scan_tmp[wave] = inc;
        __syncthreads();
        std::uint32_t wave_off = 0;
        std::uint32_t sum = 0;
        for (int w = 0; w < kWmmaWaves; ++w) {
          if (w < static_cast<int>(wave))
            wave_off += scan_tmp[w];
          sum += scan_tmp[w];
        }
        __syncthreads();
        *total = sum;
        return wave_off + inc - value;
      };
      std::uint32_t total_gt = 0;
      std::uint32_t total_eq = 0;
      const std::uint32_t gt_off = block_scan(gt, &total_gt);
      const std::uint32_t eq_off = block_scan(eq, &total_eq);
      const std::uint32_t eq_take =
          top_blocks > total_gt ? top_blocks - total_gt : 0;
      const std::uint32_t my_eq =
          eq_off < eq_take ? min(eq, eq_take - eq_off) : 0;
      std::uint32_t sel_total = 0;
      const std::uint32_t sel_off = block_scan(gt + my_eq, &sel_total);
      // Sinks, selected blocks in ascending order, then the local window.
      for (std::uint32_t b = tid; b < kSparseSink; b += kWmmaThreads)
        blist[b] = b;
      std::uint32_t out_at = kSparseSink + sel_off;
      std::uint32_t eq_seen = 0;
      for (std::uint32_t b = b0; b < b1; ++b) {
        const std::uint32_t sc = scores[b];
        bool take = sc > thr;
        if (!take && sc == thr && eq_seen < my_eq) {
          take = true;
          ++eq_seen;
        }
        if (take)
          blist[out_at++] = cand_first + b;
      }
      const std::uint32_t local_base = kSparseSink + sel_total;
      for (std::uint32_t b = local_first + tid; b <= local_last;
           b += kWmmaThreads)
        blist[local_base + (b - local_first)] = b;
      list_count = local_base + (local_last - local_first + 1);
    }
    __syncthreads();
    n_tiles = list_count * (kSparseBlock / kWmmaKeys);
  }
  const auto tile_key = [&](std::uint32_t j) {
    if constexpr (kSparse) {
      return blist[j / (kSparseBlock / kWmmaKeys)] * kSparseBlock +
             (j % (kSparseBlock / kWmmaKeys)) * kWmmaKeys;
    } else {
      return first_key + j * kWmmaKeys;
    }
  };

  constexpr int kLoads = kWmmaKeys * 16 / kWmmaThreads;
  uint4 kpre[kLoads];
  uint4 vpre[kLoads];
  // Lanes run along keys: the V^T stores of a wave then hit consecutive words.
  const auto fetch = [&](std::uint32_t kt) {
#pragma unroll
    for (int u = 0; u < kLoads; ++u) {
      const std::uint32_t idx = tid + u * kWmmaThreads;
      const std::uint32_t key = idx % kWmmaKeys;
      const std::uint32_t c = idx / kWmmaKeys;
      kpre[u] = make_uint4(0, 0, 0, 0);
      vpre[u] = make_uint4(0, 0, 0, 0);
      if (kt + key <= pos_max) {
        const std::size_t src =
            (kt + key) * kv_row + kv_head * kWmmaHeadDim + c * 8;
        kpre[u] = *reinterpret_cast<const uint4*>(k_cache + src);
        vpre[u] = *reinterpret_cast<const uint4*>(v_cache + src);
      }
    }
  };
  const auto stage = [&]() {
    const bool odd = (lane & 1U) != 0;
#pragma unroll
    for (int u = 0; u < kLoads; ++u) {
      const std::uint32_t idx = tid + u * kWmmaThreads;
      const std::uint32_t key = idx % kWmmaKeys;
      const std::uint32_t c = idx / kWmmaKeys;
      *reinterpret_cast<uint4*>(k_s + key * kWmmaKStride + c * 8) = kpre[u];
      const std::uint32_t words[4] = {vpre[u].x, vpre[u].y, vpre[u].z,
                                      vpre[u].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // Dims 2j and 2j + 1 of this key; the lane pair swaps halves so each
        // lane stores one 32-bit word holding two neighbouring keys.
        const std::uint32_t mine = words[j];
        const std::uint32_t other = __shfl_xor(mine, 1);
        const std::uint32_t lo_dim = odd ? (mine >> 16) : (mine & 0xffffU);
        const std::uint32_t lo_other = odd ? (other >> 16) : (other & 0xffffU);
        const std::uint32_t dim = c * 8 + 2 * j + (odd ? 1U : 0U);
        const std::uint32_t pair =
            odd ? (lo_other | (lo_dim << 16)) : (lo_dim | (lo_other << 16));
        *reinterpret_cast<std::uint32_t*>(v_t + dim * kWmmaVtStride +
                                          (key & ~1U)) = pair;
      }
    }
  };
  fetch(tile_key(0));
  for (std::uint32_t j = 0; j < n_tiles; ++j) {
    const std::uint32_t kt = tile_key(j);
    stage();
    __syncthreads();
    if (j + 1 < n_tiles)
      fetch(tile_key(j + 1));

    // S^T blocks: rows are keys, columns are this wave's queries.
    v8f s_acc[kNb];
#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
      s_acc[nb] = v8f{};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const v16h k_frag =
            LoadFrag(k_s + (nb * 16 + sub) * kWmmaKStride + ks * 16);
        s_acc[nb] = Wmma(k_frag, q_frag[ks], s_acc[nb]);
      }
    }

    float val[kNb][8];
    float tile_max = kNegInf;
#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint32_t key = kt + nb * 16 + 2 * i + half_id;
        const bool in_window =
            window == 0 || qpos < window || key + window > qpos;
        const bool valid = live && key <= qpos && in_window;
        val[nb][i] = valid ? s_acc[nb][i] * scale : kNegInf;
        tile_max = fmaxf(tile_max, val[nb][i]);
      }
    }
    tile_max = fmaxf(tile_max, __shfl_xor(tile_max, 16));
    const float m_new = fmaxf(run_max, tile_max);
    const float alpha = __builtin_amdgcn_exp2f(run_max - m_new);
    float psum = 0.0F;
    v16h p_frag[kNb];
#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
      std::uint32_t own[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        const float p0 = __builtin_amdgcn_exp2f(val[nb][2 * m] - m_new);
        const float p1 = __builtin_amdgcn_exp2f(val[nb][2 * m + 1] - m_new);
        psum += p0 + p1;
        const __half2 h2 = __floats2half2_rn(p0, p1);
        own[m] = *reinterpret_cast<const std::uint32_t*>(&h2);
      }
      std::uint32_t partner[4];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        partner[m] = static_cast<std::uint32_t>(
            __shfl_xor(static_cast<int>(own[m]), 16));
      }
      // Keys 2i come from the lower lane half, 2i + 1 from the upper one.
      std::uint32_t dw[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        dw[i] =
            __builtin_amdgcn_perm(partner[i / 2], own[i / 2], perm_sel[i & 1]);
      }
      __builtin_memcpy(&p_frag[nb], dw, 32);
    }
    run_sum = run_sum * alpha + psum;
    run_max = m_new;
    // The running maximum rarely moves once a few tiles are in: skip the
    // rescale when no lane of the wave changed it.
    if (__builtin_amdgcn_ballot_w32(alpha != 1.0F) != 0) {
#pragma unroll
      for (int t = 0; t < 8; ++t)
        o_acc[t] *= alpha;
    }

#pragma unroll
    for (int nb = 0; nb < kNb; ++nb) {
#pragma unroll
      for (int t = 0; t < 8; ++t) {
        const v16h v_frag =
            LoadFrag(v_t + (t * 16 + sub) * kWmmaVtStride + nb * 16);
        o_acc[t] = Wmma(v_frag, p_frag[nb], o_acc[t]);
      }
    }
    __syncthreads();
  }

  if (!live)
    return;
  const float total = run_sum + __shfl_xor(run_sum, 16);
  const float inv = 1.0F / total;
  float* dst =
      out + (static_cast<std::size_t>(token) * heads + head) * kWmmaHeadDim;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
#pragma unroll
    for (int i = 0; i < 8; ++i)
      dst[t * 16 + 2 * i + half_id] = o_acc[t][i] * inv;
  }
}

__global__ void SigmoidTopKKernel(const float* logits, std::int32_t* ids,
                                  float* weights, std::uint32_t n_experts,
                                  std::uint32_t k, bool normalize,
                                  float scale) {
  const float* row = logits + static_cast<std::size_t>(blockIdx.x) * n_experts;
  const float p = threadIdx.x < n_experts
                      ? 1.0F / (1.0F + expf(-row[threadIdx.x]))
                      : kNegInf;
  SelectTopK(p, n_experts, k, ids + blockIdx.x * k, weights + blockIdx.x * k,
             normalize, scale);
}

// One wave per token: eight experts per lane, selection by shuffles only.
__global__ __launch_bounds__(128) void SigmoidTopKWaveKernel(
    const float* logits, std::int32_t* ids, float* weights,
    std::uint32_t n_tokens, std::uint32_t n_experts, std::uint32_t k,
    bool normalize, float scale) {
  constexpr int kPer = 8;
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t token =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (token >= n_tokens)
    return;
  const float* row = logits + static_cast<std::size_t>(token) * n_experts;
  float v[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const std::uint32_t e = j * warpSize + lane;
    v[j] = e < n_experts ? 1.0F / (1.0F + expf(-row[e])) : kNegInf;
  }
  float picked[16];
  for (std::uint32_t r = 0; r < k; ++r) {
    float best = v[0];
    int slot = 0;
#pragma unroll
    for (int j = 1; j < kPer; ++j) {
      if (v[j] > best) {
        best = v[j];
        slot = j;
      }
    }
    int idx = slot * warpSize + static_cast<int>(lane);
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float ov = __shfl_xor(best, off);
      const int oi = __shfl_xor(idx, off);
      if (ov > best || (ov == best && oi < idx)) {
        best = ov;
        idx = oi;
      }
    }
    if (static_cast<int>(lane) == idx % warpSize) {
#pragma unroll
      for (int j = 0; j < kPer; ++j) {
        if (j == idx / warpSize)
          v[j] = kNegInf;
      }
    }
    picked[r] = best;
    if (lane == 0)
      ids[token * k + r] = idx;
  }
  if (lane == 0) {
    float sum = 0.0F;
    for (std::uint32_t r = 0; r < k; ++r)
      sum += picked[r];
    const float denom = normalize ? fmaxf(sum, 6.103515625e-5F) : 1.0F;
    for (std::uint32_t r = 0; r < k; ++r) {
      weights[token * k + r] = picked[r] / denom * scale;
    }
  }
}

// One wave per (expert, token): lanes stream the BF16 row with 16-byte loads.
__global__ void RouterBf16Kernel(const uint4* w, const float* x, float* out,
                                 std::uint32_t n_experts, std::uint32_t k) {
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t wave = threadIdx.x / warpSize;
  const std::uint32_t waves = blockDim.x / warpSize;
  const std::uint32_t e = blockIdx.x * waves + wave;
  if (e >= n_experts)
    return;
  const std::uint32_t row_vecs = k / 8;
  const uint4* wr = w + static_cast<std::size_t>(e) * row_vecs;
  const float4* xr = reinterpret_cast<const float4*>(
      x + static_cast<std::size_t>(blockIdx.y) * k);
  float acc = 0.0F;
  for (std::uint32_t v = lane; v < row_vecs; v += warpSize) {
    const uint4 u = wr[v];
    const float4 xa = xr[2 * v];
    const float4 xb = xr[2 * v + 1];
    acc += __uint_as_float(u.x << 16) * xa.x;
    acc += __uint_as_float(u.x & 0xffff0000U) * xa.y;
    acc += __uint_as_float(u.y << 16) * xa.z;
    acc += __uint_as_float(u.y & 0xffff0000U) * xa.w;
    acc += __uint_as_float(u.z << 16) * xb.x;
    acc += __uint_as_float(u.z & 0xffff0000U) * xb.y;
    acc += __uint_as_float(u.w << 16) * xb.z;
    acc += __uint_as_float(u.w & 0xffff0000U) * xb.w;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_xor(acc, off);
  }
  if (lane == 0) {
    out[static_cast<std::size_t>(blockIdx.y) * n_experts + e] = acc;
  }
}

constexpr unsigned kRouterTokens = 16;
constexpr unsigned kRouterChunk = 128;

// Thread per expert, kRouterTokens tokens per block: each BF16 weight row is
// read once per token group, activations are staged through LDS.
__global__ void __launch_bounds__(256)
    RouterBf16WideKernel(const uint4* w, const float* x, float* out,
                         std::uint32_t n_tokens, std::uint32_t n_experts,
                         std::uint32_t k) {
  __shared__ float4 xs[kRouterTokens][kRouterChunk / 4];
  const std::uint32_t e = threadIdx.x;
  const std::uint32_t t0 = blockIdx.x * kRouterTokens;
  const std::uint32_t row_vecs = k / 8;
  const uint4* wr =
      w + static_cast<std::size_t>(e < n_experts ? e : 0) * row_vecs;
  float acc[kRouterTokens];
#pragma unroll
  for (unsigned t = 0; t < kRouterTokens; ++t)
    acc[t] = 0.0F;
  for (std::uint32_t kc = 0; kc < k; kc += kRouterChunk) {
    __syncthreads();
    for (unsigned i = threadIdx.x; i < kRouterTokens * kRouterChunk / 4;
         i += blockDim.x) {
      const unsigned t = i / (kRouterChunk / 4);
      const unsigned c = i % (kRouterChunk / 4);
      const std::uint32_t row = t0 + t < n_tokens ? t0 + t : n_tokens - 1;
      xs[t][c] = *reinterpret_cast<const float4*>(
          x + static_cast<std::size_t>(row) * k + kc + c * 4);
    }
    __syncthreads();
    for (unsigned v = 0; v < kRouterChunk / 8; ++v) {
      const uint4 u = wr[kc / 8 + v];
      const float wv[8] = {
          __uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000U),
          __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000U),
          __uint_as_float(u.z << 16), __uint_as_float(u.z & 0xffff0000U),
          __uint_as_float(u.w << 16), __uint_as_float(u.w & 0xffff0000U)};
#pragma unroll
      for (unsigned t = 0; t < kRouterTokens; ++t) {
        const float4 a = xs[t][2 * v];
        const float4 b = xs[t][2 * v + 1];
        acc[t] += wv[0] * a.x;
        acc[t] += wv[1] * a.y;
        acc[t] += wv[2] * a.z;
        acc[t] += wv[3] * a.w;
        acc[t] += wv[4] * b.x;
        acc[t] += wv[5] * b.y;
        acc[t] += wv[6] * b.z;
        acc[t] += wv[7] * b.w;
      }
    }
  }
  if (e >= n_experts)
    return;
#pragma unroll
  for (unsigned t = 0; t < kRouterTokens; ++t) {
    if (t0 + t < n_tokens) {
      out[static_cast<std::size_t>(t0 + t) * n_experts + e] = acc[t];
    }
  }
}

constexpr int kNpuTileM = 128;
constexpr int kNpuTileK = 64;
constexpr int kNpuGroupBytes = 9;

__device__ inline void NpuEncodeGroup(const float* v, std::uint8_t* out) {
  unsigned max_exp = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    max_exp = max(max_exp, (__float_as_uint(v[i]) >> 23) & 0xFFU);
  const float inv = __uint_as_float(max_exp >= 7 ? (260U - max_exp) << 23 : 0U);
  out[0] = static_cast<std::uint8_t>(max_exp);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float q = fminf(127.0F, fmaxf(-128.0F, rintf(v[i] * inv)));
    out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
  }
}

// The part of the group the high encoding cannot hold, encoded alone.
__device__ inline void NpuEncodeResidual(const float* v, std::uint8_t* out) {
  unsigned max_exp = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i)
    max_exp = max(max_exp, (__float_as_uint(v[i]) >> 23) & 0xFFU);
  const float inv = __uint_as_float(max_exp >= 7 ? (260U - max_exp) << 23 : 0U);
  const float unit = __uint_as_float(max_exp >= 7 ? (max_exp - 6U) << 23 : 0U);
  float r[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float q = fminf(127.0F, fmaxf(-128.0F, rintf(v[i] * inv)));
    r[i] = v[i] - q * unit;
  }
  NpuEncodeGroup(r, out);
}

// A tiles (128 x 64) of the float activations, both the high and the residual
// stream in one pass, in the layout of the GEMM's L1 tiles.
__global__ void __launch_bounds__(256)
    NpuEncodeBothKernel(const float* __restrict__ x,
                        std::uint8_t* __restrict__ hi,
                        std::uint8_t* __restrict__ lo, std::uint32_t k_slice,
                        std::size_t ldx) {
  __shared__ __align__(16) float tile[kNpuTileM * kNpuTileK];
  __shared__ __align__(16)
      std::uint8_t staged_hi[kNpuTileM * kNpuTileK / 8 * kNpuGroupBytes];
  __shared__ __align__(16)
      std::uint8_t staged_lo[kNpuTileM * kNpuTileK / 8 * kNpuGroupBytes];
  const std::uint32_t tiles_k = k_slice / kNpuTileK;
  const std::uint32_t tile_row = blockIdx.x / tiles_k;
  const std::uint32_t tile_col = blockIdx.x % tiles_k;
  const std::uint32_t t = threadIdx.x;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const std::uint32_t row = (t >> 3) + 32 * j;
    const std::uint32_t vec = t & 7;
    const float* src = x + (std::size_t{tile_row} * kNpuTileM + row) * ldx +
                       tile_col * kNpuTileK + vec * 8;
    float4* dst = reinterpret_cast<float4*>(&tile[row * kNpuTileK + vec * 8]);
    dst[0] = *reinterpret_cast<const float4*>(src);
    dst[1] = *reinterpret_cast<const float4*>(src + 4);
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const std::uint32_t gi = t + 256 * j;
    const std::uint32_t rib = gi & 7;
    const std::uint32_t block = (gi >> 3) & 1;
    const std::uint32_t sc = (gi >> 4) & 7;
    const std::uint32_t sr = gi >> 7;
    const std::uint32_t row = sr * 16 + block * 8 + rib;
    float v[8];
    const float4* g =
        reinterpret_cast<const float4*>(&tile[row * kNpuTileK + sc * 8]);
    *reinterpret_cast<float4*>(v) = g[0];
    *reinterpret_cast<float4*>(v + 4) = g[1];
    NpuEncodeGroup(v, staged_hi + gi * kNpuGroupBytes);
    NpuEncodeResidual(v, staged_lo + gi * kNpuGroupBytes);
  }
  __syncthreads();
  constexpr std::uint32_t kTileBytes =
      kNpuTileM * kNpuTileK / 8 * kNpuGroupBytes;
  const std::size_t base = std::size_t{blockIdx.x} * kTileBytes;
  for (std::uint32_t i = t; i < kTileBytes / 16; i += 256) {
    reinterpret_cast<uint4*>(hi + base)[i] =
        reinterpret_cast<const uint4*>(staged_hi)[i];
    reinterpret_cast<uint4*>(lo + base)[i] =
        reinterpret_cast<const uint4*>(staged_lo)[i];
  }
}

// Block-float C streams of the NPU GEMM: 8 values share one exponent byte
// (9 bytes per group), laid out in 512 x 128 tiles.
constexpr std::uint32_t kNpuTileRowsC = 512;
constexpr std::uint32_t kNpuTileN = 128;

// silu(gate) * up straight from the high and residual C streams of the shared
// expert's gate and up GEMMs, written as F16 rows of valid_cols times `scale`
// (the caller folds 1 / scale back in after the down projection), so tokens
// with very large activations stay inside the F16 range.
__global__ void NpuSwigluScaledKernel(const std::uint8_t* __restrict__ gate_hi,
                                      const std::uint8_t* __restrict__ up_hi,
                                      const std::uint8_t* __restrict__ gate_lo,
                                      const std::uint8_t* __restrict__ up_lo,
                                      __half* __restrict__ out,
                                      std::uint32_t tokens, std::uint32_t cols,
                                      std::uint32_t valid_cols, float scale) {
  const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (g >= std::size_t{tokens} * cols / 8)
    return;
  constexpr std::uint32_t kGroupsPerTile = kNpuTileRowsC * kNpuTileN / 8;
  const std::size_t tile = g / kGroupsPerTile;
  const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
  const std::uint32_t tiles_n = cols / kNpuTileN;
  const std::uint32_t tile_row = static_cast<std::uint32_t>(tile / tiles_n);
  const std::uint32_t tile_col = static_cast<std::uint32_t>(tile % tiles_n);
  const std::uint32_t rib = gi & 7;
  const std::uint32_t bc = (gi >> 3) & 1;
  const std::uint32_t br = (gi >> 4) & 1;
  const std::uint32_t sc = (gi >> 5) & 7;
  const std::uint32_t sr = gi >> 8;
  const std::size_t row =
      std::size_t{tile_row} * kNpuTileRowsC + (sr * 2 + br) * 8 + rib;
  const std::uint32_t col = tile_col * kNpuTileN + (sc * 2 + bc) * 8;
  if (col >= valid_cols)
    return;
  const std::uint8_t* pg = gate_hi + g * 9;
  const std::uint8_t* pu = up_hi + g * 9;
  const std::uint8_t* pgl = gate_lo + g * 9;
  const std::uint8_t* pul = up_lo + g * 9;
  const float mg = ldexpf(1.0F, static_cast<int>(pg[0]) - 127) / 64.0F;
  const float mu = ldexpf(1.0F, static_cast<int>(pu[0]) - 127) / 64.0F;
  const float mgl = ldexpf(1.0F, static_cast<int>(pgl[0]) - 127) / 64.0F;
  const float mul = ldexpf(1.0F, static_cast<int>(pul[0]) - 127) / 64.0F;
  __half2 packed[4];
#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    float v[2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float gv =
          static_cast<float>(static_cast<std::int8_t>(pg[1 + i + e])) * mg +
          static_cast<float>(static_cast<std::int8_t>(pgl[1 + i + e])) * mgl;
      const float uv =
          static_cast<float>(static_cast<std::int8_t>(pu[1 + i + e])) * mu +
          static_cast<float>(static_cast<std::int8_t>(pul[1 + i + e])) * mul;
      v[e] = gv / (1.0F + expf(-gv)) * uv * scale;
    }
    packed[i / 2] = __floats2half2_rn(v[0], v[1]);
  }
  *reinterpret_cast<uint4*>(out + row * valid_cols + col) =
      *reinterpret_cast<const uint4*>(packed);
}

// High and low (residual) pass of one slice decoded and summed in one write.
__global__ void NpuDecodeFusedKernel(NpuStreams s, float* __restrict__ out,
                                     std::uint32_t tokens, std::uint32_t cols,
                                     std::size_t pitch) {
  const std::size_t g = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (g >= std::size_t{tokens} * cols / 8)
    return;
  constexpr std::uint32_t kGroupsPerTile = kNpuTileRowsC * kNpuTileN / 8;
  const std::size_t tile = g / kGroupsPerTile;
  const std::uint32_t gi = static_cast<std::uint32_t>(g % kGroupsPerTile);
  const std::uint32_t tiles_n = cols / kNpuTileN;
  const std::uint32_t tile_row = static_cast<std::uint32_t>(tile / tiles_n);
  const std::uint32_t tile_col = static_cast<std::uint32_t>(tile % tiles_n);
  const std::uint32_t rib = gi & 7;
  const std::uint32_t bc = (gi >> 3) & 1;
  const std::uint32_t br = (gi >> 4) & 1;
  const std::uint32_t sc = (gi >> 5) & 7;
  const std::uint32_t sr = gi >> 8;
  const std::size_t row =
      std::size_t{tile_row} * kNpuTileRowsC + (sr * 2 + br) * 8 + rib;
  const std::uint32_t col = tile_col * kNpuTileN + (sc * 2 + bc) * 8;
  const std::uint8_t* hi = s.hi[blockIdx.y] + g * 9;
  const std::uint8_t* lo = s.lo[blockIdx.y] + g * 9;
  const float mult_hi = ldexpf(1.0F, static_cast<int>(hi[0]) - 127) / 64.0F;
  const float mult_lo = ldexpf(1.0F, static_cast<int>(lo[0]) - 127) / 64.0F;
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    v[i] = static_cast<float>(static_cast<std::int8_t>(hi[1 + i])) * mult_hi +
           static_cast<float>(static_cast<std::int8_t>(lo[1 + i])) * mult_lo;
  }
  float* dst = out + row * pitch + blockIdx.y * cols + col;
  *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
  *reinterpret_cast<float4*>(dst + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float g = gate[i];
  gate[i] = g / (1.0F + expf(-g)) * up[i];
}

// Largest |value| a row may keep: a Q8_1 block of 32 then sums to at most
// 32768, inside the half range of its sum field.
constexpr float kSwigluLimit = 1024.0F;

// One block of cols / 4 threads per row, one float4 each.
__global__ void SwigluScaledKernel(float* gate, const float* up, float* scale,
                                   std::uint32_t cols) {
  __shared__ float red[kBlock / 32];
  const std::size_t base =
      static_cast<std::size_t>(blockIdx.x) * cols + threadIdx.x * 4;
  const float4 g = *reinterpret_cast<const float4*>(gate + base);
  const float4 u = *reinterpret_cast<const float4*>(up + base);
  float v[4] = {
      g.x / (1.0F + expf(-g.x)) * u.x, g.y / (1.0F + expf(-g.y)) * u.y,
      g.z / (1.0F + expf(-g.z)) * u.z, g.w / (1.0F + expf(-g.w)) * u.w};
  float amax =
      fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off));
  }
  if (threadIdx.x % warpSize == 0)
    red[threadIdx.x / warpSize] = amax;
  __syncthreads();
  float row_max = 0.0F;
  for (unsigned w = 0; w < blockDim.x / warpSize; ++w) {
    row_max = fmaxf(row_max, red[w]);
  }
  const float s = row_max > kSwigluLimit
                      ? exp2f(ceilf(log2f(row_max / kSwigluLimit)))
                      : 1.0F;
  *reinterpret_cast<float4*>(gate + base) =
      make_float4(v[0] / s, v[1] / s, v[2] / s, v[3] / s);
  if (threadIdx.x == 0)
    scale[blockIdx.x] = s;
}

__global__ void MoeCombineKernel(float* x, const float* expert_out,
                                 const float* weights, const float* shared,
                                 std::uint32_t k, std::uint32_t dim) {
  const std::size_t t = blockIdx.y;
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < dim;
       i += gridDim.x * blockDim.x) {
    float acc = shared[t * dim + i];
    for (std::uint32_t s = 0; s < k; ++s) {
      acc += weights[t * k + s] * expert_out[(t * k + s) * dim + i];
    }
    x[t * dim + i] += acc;
  }
}

__global__ void NarrowRowsKernel(const float* in, __half* out, std::uint32_t k,
                                 std::uint32_t ldx) {
  const std::size_t row = blockIdx.y;
  for (std::uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < k;
       i += gridDim.x * blockDim.x * 4) {
    const float4 v = *reinterpret_cast<const float4*>(in + row * k + i);
    __half2* dst = reinterpret_cast<__half2*>(out + row * ldx + i);
    dst[0] = __floats2half2_rn(v.x, v.y);
    dst[1] = __floats2half2_rn(v.z, v.w);
  }
}

__global__ void ExpandQ8Kernel(const unsigned char* q8, __half* out,
                               std::size_t blocks) {
  const std::size_t b =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks)
    return;
  const unsigned char* src = q8 + b * kQ8Bytes;
  const float d = __half2float(*reinterpret_cast<const __half*>(src));
  for (int i = 0; i < kQ8Block; ++i) {
    out[b * kQ8Block + i] = __float2half(
        d * static_cast<float>(static_cast<signed char>(src[2 + i])));
  }
}

__global__ void AddKernel(float* x, const float* y, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    x[i] += y[i];
}

}  // namespace

void EmbedQ8(const void* table, const std::int32_t* tokens, float* out,
             std::uint32_t n_tokens, std::uint32_t hidden, hipStream_t stream) {
  EmbedQ8Kernel<<<n_tokens, kBlock, 0, stream>>>(
      static_cast<const unsigned char*>(table), tokens, out, hidden);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  RmsNormKernel<<<dim3(n_rows, groups), kBlock, 0, stream>>>(x, gamma, out, dim,
                                                             groups, eps);
}

void RopeNeoX(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t head_dim, const std::uint32_t* start_pos,
              float base, hipStream_t stream) {
  RopeKernel<<<dim3(Blocks(std::size_t{heads} * head_dim / 2), n_tokens),
               kBlock, 0, stream>>>(x, heads, head_dim, start_pos, base);
}

void StoreKv(const float* k, const float* v, __half* k_cache, __half* v_cache,
             std::uint32_t n_tokens, std::uint32_t row_dim,
             const std::uint32_t* start_pos, hipStream_t stream) {
  StoreKvKernel<<<dim3(Blocks(row_dim), n_tokens), kBlock, 0, stream>>>(
      k, v, k_cache, v_cache, row_dim, start_pos);
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float* out, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t head_dim,
               std::uint32_t window, hipStream_t stream) {
  AttentionKernel<<<dim3(heads, n_tokens), kAttnBlock, 0, stream>>>(
      q, k_cache, v_cache, sinks, out, start_pos, heads, kv_heads, head_dim,
      window);
}

bool AttentionWmma(const __half* q, const __half* k_cache,
                   const __half* v_cache, const float* sinks, float* out,
                   std::uint32_t n_tokens, const std::uint32_t* start_pos,
                   std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t head_dim, std::uint32_t window,
                   hipStream_t stream) {
  if (head_dim != kWmmaHeadDim || heads != kv_heads * kWmmaGroup)
    return false;
  AttentionWmmaKernel<false>
      <<<dim3(kv_heads, (n_tokens + kWmmaTokens - 1) / kWmmaTokens),
         kWmmaThreads, 0, stream>>>(q, k_cache, v_cache, sinks, out, n_tokens,
                                    start_pos, heads, kv_heads, window, nullptr,
                                    0, 0);
  return true;
}

bool AttentionWmmaSparse(const __half* q, const __half* k_cache,
                         const __half* v_cache, const __half* kmm,
                         std::uint32_t nblk, std::uint32_t top_blocks,
                         const float* sinks, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t head_dim,
                         hipStream_t stream) {
  if (head_dim != kWmmaHeadDim || heads != kv_heads * kWmmaGroup ||
      nblk > 4096 || top_blocks + kSparseSink + kSparseLocal + 4 > kSparseList)
    return false;
  AttentionWmmaKernel<true>
      <<<dim3(kv_heads, (n_tokens + kWmmaTokens - 1) / kWmmaTokens),
         kWmmaThreads, 0, stream>>>(q, k_cache, v_cache, sinks, out, n_tokens,
                                    start_pos, heads, kv_heads, 0, kmm, nblk,
                                    top_blocks);
  return true;
}

void PoolKeyBlocks(const __half* k_cache, __half* kmm,
                   const std::uint32_t* start_pos, std::uint32_t start_host,
                   std::uint32_t n_tokens, std::uint32_t kv_heads,
                   std::uint32_t nblk, hipStream_t stream) {
  const std::uint32_t first = start_host / kSparseBlock;
  const std::uint32_t last = (start_host + n_tokens - 1) / kSparseBlock;
  PoolKeyBlocksKernel<<<dim3(last - first + 1, kv_heads), kWmmaHeadDim, 0,
                        stream>>>(k_cache, kmm, start_pos, n_tokens, kv_heads,
                                  nblk);
}

void AttentionSplit(const float* q, const __half* k_cache,
                    const __half* v_cache, const float* sinks, float* out,
                    void* out_q8, float* partial, std::uint32_t n_tokens,
                    const std::uint32_t* start_pos, std::uint32_t heads,
                    std::uint32_t kv_heads, std::uint32_t head_dim,
                    std::uint32_t window, std::uint32_t splits,
                    hipStream_t stream) {
  AttentionSliceKernel<<<dim3(kv_heads, splits, n_tokens),
                         (heads / kv_heads) * 32, 0, stream>>>(
      q, k_cache, v_cache, partial, start_pos, heads, kv_heads, head_dim,
      window, splits);
  AttentionMergeKernel<<<dim3(heads, n_tokens), kAttnBlock, 0, stream>>>(
      partial, sinks, out, static_cast<unsigned char*>(out_q8), heads, head_dim,
      splits);
}

void ResidualNormQuant(float* x, const float* delta, const float* gamma,
                       float* xn, void* q8, std::uint32_t n_tokens,
                       std::uint32_t dim, float eps, hipStream_t stream) {
  ResidualNormQuantKernel<<<n_tokens, dim / 8, 0, stream>>>(
      x, delta, gamma, xn, static_cast<unsigned char*>(q8), dim, eps);
}

void MoeCombineNormQuant(float* x, const float* expert_out,
                         const float* weights, const float* expert_scale,
                         const float* shared, const float* shared_scale,
                         std::uint32_t k, const float* gamma, float* xn,
                         void* q8, std::uint32_t n_tokens, std::uint32_t dim,
                         float eps, hipStream_t stream) {
  MoeCombineNormQuantKernel<<<n_tokens, dim / 8, 0, stream>>>(
      x, expert_out, weights, expert_scale, shared, shared_scale, k, gamma, xn,
      static_cast<unsigned char*>(q8), dim, eps);
}

void SwigluScaled(float* gate, const float* up, float* scale,
                  std::uint32_t rows, std::uint32_t cols, hipStream_t stream) {
  SwigluScaledKernel<<<rows, cols / 4, 0, stream>>>(gate, up, scale, cols);
}

void QkvPrepare(const QkvSources& src, const float* q_gamma,
                const float* k_gamma, float* q_out, __half* k_cache,
                __half* v_cache, std::uint32_t n_tokens, std::uint32_t heads,
                std::uint32_t kv_heads, std::uint32_t head_dim,
                const std::uint32_t* start_pos, float base, float eps,
                bool rope, hipStream_t stream, __half* q_half) {
  QkvPrepareKernel<<<dim3(heads + 2 * kv_heads, n_tokens), head_dim, 0,
                     stream>>>(src, q_gamma, k_gamma, q_out, k_cache, v_cache,
                               heads, kv_heads, head_dim, start_pos, base, eps,
                               rope, q_half);
}

void SigmoidTopK(const float* logits, std::int32_t* ids, float* weights,
                 std::uint32_t n_tokens, std::uint32_t n_experts,
                 std::uint32_t k, bool normalize, float scale,
                 hipStream_t stream) {
  if (n_tokens > 8 && n_experts <= 256 && k <= 16) {
    SigmoidTopKWaveKernel<<<(n_tokens + 3) / 4, 128, 0, stream>>>(
        logits, ids, weights, n_tokens, n_experts, k, normalize, scale);
    return;
  }
  SigmoidTopKKernel<<<n_tokens, kBlock, 0, stream>>>(
      logits, ids, weights, n_experts, k, normalize, scale);
}

void RouterBf16(const void* w, const float* x, float* out,
                std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream) {
  if (n_tokens > 8 && n_experts <= 256 && k % kRouterChunk == 0) {
    RouterBf16WideKernel<<<(n_tokens + kRouterTokens - 1) / kRouterTokens, 256,
                           0, stream>>>(static_cast<const uint4*>(w), x, out,
                                        n_tokens, n_experts, k);
    return;
  }
  constexpr unsigned kWaves = kBlock / 32;
  RouterBf16Kernel<<<dim3((n_experts + kWaves - 1) / kWaves, n_tokens), kBlock,
                     0, stream>>>(static_cast<const uint4*>(w), x, out,
                                  n_experts, k);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  SwigluKernel<<<Blocks(count), kBlock, 0, stream>>>(gate, up, count);
}

void MoeCombine(float* x, const float* expert_out, const float* weights,
                const float* shared, std::uint32_t n_tokens, std::uint32_t k,
                std::uint32_t dim, hipStream_t stream) {
  MoeCombineKernel<<<dim3(Blocks(dim), n_tokens), kBlock, 0, stream>>>(
      x, expert_out, weights, shared, k, dim);
}

void NarrowRows(const float* in, __half* out, std::uint32_t n_rows,
                std::uint32_t k, std::uint32_t ldx, hipStream_t stream) {
  NarrowRowsKernel<<<dim3(Blocks(k / 4), n_rows), kBlock, 0, stream>>>(in, out,
                                                                       k, ldx);
}

void ExpandQ8ToHalf(const void* q8, __half* out, std::size_t count,
                    hipStream_t stream) {
  const std::size_t blocks = count / kQ8Block;
  ExpandQ8Kernel<<<Blocks(blocks), kBlock, 0, stream>>>(
      static_cast<const unsigned char*>(q8), out, blocks);
}

void AddInPlace(float* x, const float* y, std::size_t count,
                hipStream_t stream) {
  AddKernel<<<Blocks(count), kBlock, 0, stream>>>(x, y, count);
}

}  // namespace gufo::models::flint::gpu::rocm

namespace gufo::models::flint::gpu::rocm {

void NpuSwigluScaled(const void* gate_hi, const void* up_hi,
                     const void* gate_lo, const void* up_lo, __half* out,
                     std::uint32_t tokens, std::uint32_t cols,
                     std::uint32_t valid_cols, float scale,
                     hipStream_t stream) {
  const std::size_t groups = std::size_t{tokens} * cols / 8;
  NpuSwigluScaledKernel<<<(groups + 255) / 256, 256, 0, stream>>>(
      static_cast<const std::uint8_t*>(gate_hi),
      static_cast<const std::uint8_t*>(up_hi),
      static_cast<const std::uint8_t*>(gate_lo),
      static_cast<const std::uint8_t*>(up_lo), out, tokens, cols, valid_cols,
      scale);
}

void NpuEncodeBoth(const float* x, std::size_t ldx, std::uint32_t tokens,
                   std::uint32_t k_slice, void* hi, void* lo,
                   hipStream_t stream) {
  NpuEncodeBothKernel<<<(tokens / kNpuTileM) * (k_slice / kNpuTileK), 256, 0,
                        stream>>>(x, static_cast<std::uint8_t*>(hi),
                                  static_cast<std::uint8_t*>(lo), k_slice, ldx);
}

void NpuDecodeFused(const NpuStreams& streams, std::uint32_t slices, float* out,
                    std::uint32_t tokens, std::uint32_t cols, std::size_t pitch,
                    hipStream_t stream) {
  const std::size_t groups = std::size_t{tokens} * cols / 8;
  NpuDecodeFusedKernel<<<dim3((groups + 255) / 256, slices), 256, 0, stream>>>(
      streams, out, tokens, cols, pitch);
}

}  // namespace gufo::models::flint::gpu::rocm
