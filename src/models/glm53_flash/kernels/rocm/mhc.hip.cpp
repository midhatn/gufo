#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"

#include "src/models/glm53_flash/kernels/rocm/xor_lane.hpp"

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kThreads = 1024;
constexpr int kStreamsFlat = kHcStreams * kHcHidden;
constexpr int kPerThread = kHcHidden / kThreads;
constexpr int kReduceSlots = kHcMixes + 1;
constexpr int kMaxWaves = kThreads / 32;
constexpr float kRmsEps = 1e-5F;
constexpr int kSinkhornRounds = 19;

__device__ float Sigmoid(float v) {
  return 1.0F / (1.0F + expf(-v));
}

// Sums v across the block; every thread receives the total.
__device__ float BlockSum(float v, float* scratch) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  const int wave = threadIdx.x / warpSize;
  const int waves = blockDim.x / warpSize;
  __syncthreads();
  if ((threadIdx.x % warpSize) == 0) {
    scratch[wave] = v;
  }
  __syncthreads();
  float total = 0.0F;
  for (int w = 0; w < waves; ++w) {
    total += scratch[w];
  }
  return total;
}

constexpr int kMixStride = kReduceSlots;
constexpr int kPreStride = kReduceSlots + kHcStreams;

// mixes[j] = fn_j . x and mixes[kHcMixes] = sum x^2 for one token.
template<typename X>
__global__ __launch_bounds__(kThreads) void HcMixKernel(
    const X* __restrict__ x, const float* __restrict__ fn,
    float* __restrict__ scratch_out) {
  x += static_cast<std::size_t>(blockIdx.x) * kStreamsFlat;
  scratch_out += static_cast<std::size_t>(blockIdx.x) * kPreStride;
  __shared__ float partial[kReduceSlots][kMaxWaves];
  const int tid = threadIdx.x;
  const int wave = tid / warpSize;
  const int waves = blockDim.x / warpSize;
  float acc[kReduceSlots];
  for (int j = 0; j < kReduceSlots; ++j) {
    acc[j] = 0.0F;
  }
  for (int i = tid; i < kStreamsFlat; i += kThreads) {
    const float xv = static_cast<float>(x[i]);
    acc[kHcMixes] += xv * xv;
    for (int j = 0; j < kHcMixes; ++j) {
      acc[j] += fn[static_cast<std::size_t>(j) * kStreamsFlat + i] * xv;
    }
  }
  for (int j = 0; j < kReduceSlots; ++j) {
    float v = acc[j];
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      v += __shfl_xor(v, offset);
    }
    if ((tid % warpSize) == 0) {
      partial[j][wave] = v;
    }
  }
  __syncthreads();
  if (tid < kReduceSlots) {
    float total = 0.0F;
    for (int w = 0; w < waves; ++w) {
      total += partial[tid][w];
    }
    scratch_out[tid] = total;
  }
}

// One wave per token: the post / comb / pre mixing weights. Lane i < 16 holds
// comb[i] (dst i % 4, src i / 4); the row softmax and the Sinkhorn sums are
// lane exchanges within groups of four.
__global__ __launch_bounds__(32) void HcSinkKernel(
    float* __restrict__ scratch_io, const float* __restrict__ scale,
    const float* __restrict__ base, float eps, float* __restrict__ post_out,
    float* __restrict__ comb_out, int tokens) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x;
  if (token >= tokens) {
    return;
  }
  float* scratch = scratch_io + static_cast<std::size_t>(token) * kPreStride;
  const float inv = rsqrtf(scratch[kHcMixes] / kStreamsFlat + kRmsEps);
  if (lane < kHcStreams) {
    scratch[kReduceSlots + lane] =
        Sigmoid(scratch[lane] * inv * scale[0] + base[lane]) + eps;
  } else if (lane < 2 * kHcStreams) {
    const int s = lane - kHcStreams;
    post_out[token * kHcStreams + s] =
        Sigmoid(scratch[kHcStreams + s] * inv * scale[1] +
                base[kHcStreams + s]) *
        2.0F;
  }
  const int mix = lane & 15;
  float c = scratch[2 * kHcStreams + mix] * inv * scale[2] +
            base[2 * kHcStreams + mix];
  float mx = fmaxf(c, XorLane(c, 1));
  mx = fmaxf(mx, XorLane(mx, 2));
  c = static_cast<float>(exp(static_cast<double>(c) - static_cast<double>(mx)));
  double sum = static_cast<double>(c);
  sum += XorLane(sum, 1);
  sum += XorLane(sum, 2);
  c /= static_cast<float>(sum + eps);
  for (int round = 0; round <= kSinkhornRounds * 2; ++round) {
    const bool over_src = round == 0 || (round % 2) == 0;
    double total = static_cast<double>(c);
    if (over_src) {
      total += XorLane(total, 4);
      total += XorLane(total, 8);
    } else {
      total += XorLane(total, 1);
      total += XorLane(total, 2);
    }
    c /= static_cast<float>(total + eps);
  }
  if (lane < kHcStreams * kHcStreams) {
    comb_out[static_cast<std::size_t>(token) * kHcStreams * kHcStreams + lane] =
        c;
  }
}

template<typename X>
__global__ __launch_bounds__(kThreads) void HcApplyKernel(
    const X* __restrict__ x, const float* __restrict__ scratch_in,
    const float* __restrict__ norm_w, float* __restrict__ y,
    _Float16* __restrict__ y_half) {
  x += static_cast<std::size_t>(blockIdx.x) * kStreamsFlat;
  y += static_cast<std::size_t>(blockIdx.x) * kHcHidden;
  if (y_half != nullptr) {
    y_half += static_cast<std::size_t>(blockIdx.x) * kHcHidden;
  }
  const float* pre = scratch_in +
                     static_cast<std::size_t>(blockIdx.x) * kPreStride +
                     kReduceSlots;
  __shared__ float scratch[kMaxWaves];
  const int tid = threadIdx.x;
  float z[kPerThread];
  float sq = 0.0F;
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    float v = 0.0F;
    for (int s = 0; s < kHcStreams; ++s) {
      v += static_cast<float>(x[s * kHcHidden + i]) * pre[s];
    }
    z[k] = v;
    sq += v * v;
  }
  const float inv = rsqrtf(BlockSum(sq, scratch) / kHcHidden + kRmsEps);
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    const float value = z[k] * inv * norm_w[i];
    y[i] = value;
    if (y_half != nullptr) {
      y_half[i] = static_cast<_Float16>(value);
    }
  }
}

template<typename X>
__global__ void HcPostKernel(const X* __restrict__ x,
                             const float* __restrict__ block,
                             const float* __restrict__ post,
                             const float* __restrict__ comb,
                             X* __restrict__ next) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kHcHidden) {
    return;
  }
  const int token = blockIdx.y;
  x += static_cast<std::size_t>(token) * kStreamsFlat;
  next += static_cast<std::size_t>(token) * kStreamsFlat;
  block += static_cast<std::size_t>(token) * kHcHidden;
  post += token * kHcStreams;
  comb += token * kHcStreams * kHcStreams;
  const float b = block[i];
  float xs[kHcStreams];
  for (int s = 0; s < kHcStreams; ++s) {
    xs[s] = static_cast<float>(x[s * kHcHidden + i]);
  }
  for (int d = 0; d < kHcStreams; ++d) {
    float acc = b * post[d];
    for (int s = 0; s < kHcStreams; ++s) {
      acc += xs[s] * comb[d + kHcStreams * s];
    }
    next[d * kHcHidden + i] = static_cast<X>(acc);
  }
}

template<typename X>
__global__ __launch_bounds__(kThreads) void HeadNormKernel(
    const X* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out) {
  __shared__ float scratch[kMaxWaves];
  x += static_cast<std::size_t>(blockIdx.x) * kStreamsFlat;
  out += static_cast<std::size_t>(blockIdx.x) * kHcHidden;
  const int tid = threadIdx.x;
  float v[kPerThread];
  float sq = 0.0F;
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    float acc = 0.0F;
    for (int s = 0; s < kHcStreams; ++s) {
      acc += static_cast<float>(x[s * kHcHidden + i]);
    }
    v[k] = acc / kHcStreams;
    sq += v[k] * v[k];
  }
  const float inv = rsqrtf(BlockSum(sq, scratch) / kHcHidden + kRmsEps);
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    out[i] = v[k] * inv * w[i];
  }
}

template<typename X>
__global__ void MeanStreamsKernel(const X* __restrict__ x, int first_row,
                                  float* __restrict__ dst, long long dst_stride,
                                  long long tap_offset) {
  const int r = blockIdx.x;
  const X* row = x + static_cast<std::size_t>(first_row + r) * kStreamsFlat;
  float* out = dst + r * dst_stride + tap_offset;
  for (int i = threadIdx.x; i < kHcHidden; i += blockDim.x) {
    float acc = 0.0F;
    for (int s = 0; s < kHcStreams; ++s) {
      acc += static_cast<float>(row[s * kHcHidden + i]);
    }
    out[i] = acc / kHcStreams;
  }
}

constexpr int kArgmaxThreads = 1024;

__global__ __launch_bounds__(kArgmaxThreads) void ArgmaxRowsKernel(
    const float* __restrict__ logits, int vocab, int* __restrict__ out) {
  __shared__ float best_value[kArgmaxThreads];
  __shared__ int best_index[kArgmaxThreads];
  const float* row = logits + static_cast<std::size_t>(blockIdx.x) * vocab;
  float value = -INFINITY;
  int index = 0x7FFFFFFF;
  for (int i = threadIdx.x; i < vocab; i += kArgmaxThreads) {
    if (row[i] > value) {
      value = row[i];
      index = i;
    }
  }
  best_value[threadIdx.x] = value;
  best_index[threadIdx.x] = index;
  __syncthreads();
  for (int stride = kArgmaxThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float ov = best_value[threadIdx.x + stride];
      const int oi = best_index[threadIdx.x + stride];
      if (ov > best_value[threadIdx.x] ||
          (ov == best_value[threadIdx.x] && oi < best_index[threadIdx.x])) {
        best_value[threadIdx.x] = ov;
        best_index[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[blockIdx.x] = best_index[0];
  }
}

}  // namespace

int ArgmaxRows(const float* logits, int rows, int vocab, int* out,
               hipStream_t stream) {
  ArgmaxRowsKernel<<<rows, kArgmaxThreads, 0, stream>>>(logits, vocab, out);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

namespace {

template<typename X>
int HcPreImpl(const X* x, const float* fn, const float* scale,
              const float* base, const float* norm_w, float hc_eps, float* y,
              float* post, float* comb, float* scratch, hipStream_t stream,
              int tokens, _Float16* y_half) {
  HcMixKernel<X><<<tokens, kThreads, 0, stream>>>(x, fn, scratch);
  HcSinkKernel<<<tokens, 32, 0, stream>>>(scratch, scale, base, hc_eps, post,
                                          comb, tokens);
  HcApplyKernel<X>
      <<<tokens, kThreads, 0, stream>>>(x, scratch, norm_w, y, y_half);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

template<typename X>
int HcPostImpl(const X* x, const float* block, const float* post,
               const float* comb, X* next, hipStream_t stream, int tokens) {
  constexpr int kBlock = 256;
  HcPostKernel<X>
      <<<dim3((kHcHidden + kBlock - 1) / kBlock, tokens), kBlock, 0, stream>>>(
          x, block, post, comb, next);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

template<typename X>
int HeadNormImpl(const X* x, const float* w, float* out, hipStream_t stream,
                 int rows) {
  HeadNormKernel<X><<<rows, kThreads, 0, stream>>>(x, w, out);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

template<typename X>
int MeanStreamsImpl(const X* x, int first_row, int rows, float* dst,
                    long long dst_stride, long long tap_offset,
                    hipStream_t stream) {
  MeanStreamsKernel<X>
      <<<rows, 256, 0, stream>>>(x, first_row, dst, dst_stride, tap_offset);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

__global__ void HalfToFloatKernel(const _Float16* __restrict__ in,
                                  float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<float>(in[i]);
  }
}

}  // namespace

int HcPre(const float* x, const float* fn, const float* scale,
          const float* base, const float* norm_w, float hc_eps, float* y,
          float* post, float* comb, float* scratch, hipStream_t stream,
          int tokens, std::uint16_t* y_half) {
  return HcPreImpl(x, fn, scale, base, norm_w, hc_eps, y, post, comb, scratch,
                   stream, tokens, reinterpret_cast<_Float16*>(y_half));
}

int HcPre(const std::uint16_t* x, const float* fn, const float* scale,
          const float* base, const float* norm_w, float hc_eps, float* y,
          float* post, float* comb, float* scratch, hipStream_t stream,
          int tokens, std::uint16_t* y_half) {
  return HcPreImpl(reinterpret_cast<const _Float16*>(x), fn, scale, base,
                   norm_w, hc_eps, y, post, comb, scratch, stream, tokens,
                   reinterpret_cast<_Float16*>(y_half));
}

int HcPost(const float* x, const float* block, const float* post,
           const float* comb, float* next, hipStream_t stream, int tokens) {
  return HcPostImpl(x, block, post, comb, next, stream, tokens);
}

int HcPost(const std::uint16_t* x, const float* block, const float* post,
           const float* comb, std::uint16_t* next, hipStream_t stream,
           int tokens) {
  return HcPostImpl(reinterpret_cast<const _Float16*>(x), block, post, comb,
                    reinterpret_cast<_Float16*>(next), stream, tokens);
}

int HeadNorm(const float* x, const float* w, float* out, hipStream_t stream,
             int rows) {
  return HeadNormImpl(x, w, out, stream, rows);
}

int HeadNorm(const std::uint16_t* x, const float* w, float* out,
             hipStream_t stream, int rows) {
  return HeadNormImpl(reinterpret_cast<const _Float16*>(x), w, out, stream,
                      rows);
}

int MeanStreams(const float* x, int first_row, int rows, float* dst,
                long long dst_stride, long long tap_offset,
                hipStream_t stream) {
  return MeanStreamsImpl(x, first_row, rows, dst, dst_stride, tap_offset,
                         stream);
}

int MeanStreams(const std::uint16_t* x, int first_row, int rows, float* dst,
                long long dst_stride, long long tap_offset,
                hipStream_t stream) {
  return MeanStreamsImpl(reinterpret_cast<const _Float16*>(x), first_row, rows,
                         dst, dst_stride, tap_offset, stream);
}

int HalfToFloat(const std::uint16_t* in, float* out, int n,
                hipStream_t stream) {
  HalfToFloatKernel<<<(n + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const _Float16*>(in), out, n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
