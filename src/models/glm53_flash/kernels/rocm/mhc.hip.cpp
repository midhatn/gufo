#include "src/models/glm53_flash/kernels/rocm/mhc.hpp"

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

__global__ __launch_bounds__(kThreads) void HcPreKernel(
    const float* __restrict__ x, const float* __restrict__ fn,
    const float* __restrict__ scale, const float* __restrict__ base,
    const float* __restrict__ norm_w, float eps, float* __restrict__ y,
    float* __restrict__ post_out, float* __restrict__ comb_out) {
  x += static_cast<std::size_t>(blockIdx.x) * kStreamsFlat;
  y += static_cast<std::size_t>(blockIdx.x) * kHcHidden;
  post_out += blockIdx.x * kHcStreams;
  comb_out += blockIdx.x * kHcStreams * kHcStreams;
  __shared__ float partial[kReduceSlots][kMaxWaves];
  __shared__ float mixes[kHcMixes];
  __shared__ float pre[kHcStreams];
  __shared__ float scratch[kMaxWaves];

  const int tid = threadIdx.x;
  const int wave = tid / warpSize;
  const int waves = blockDim.x / warpSize;

  float acc[kReduceSlots];
  for (int j = 0; j < kReduceSlots; ++j) {
    acc[j] = 0.0F;
  }
  for (int i = tid; i < kStreamsFlat; i += kThreads) {
    const float xv = x[i];
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
    partial[tid][0] = total;
  }
  __syncthreads();

  if (tid == 0) {
    const float inv = rsqrtf(partial[kHcMixes][0] / kStreamsFlat + kRmsEps);
    for (int j = 0; j < kHcMixes; ++j) {
      mixes[j] = partial[j][0] * inv;
    }
    float comb[kHcStreams * kHcStreams];
    for (int s = 0; s < kHcStreams; ++s) {
      pre[s] = Sigmoid(mixes[s] * scale[0] + base[s]) + eps;
      post_out[s] =
          Sigmoid(mixes[kHcStreams + s] * scale[1] + base[kHcStreams + s]) *
          2.0F;
    }
    for (int i = 0; i < kHcStreams * kHcStreams; ++i) {
      comb[i] = mixes[2 * kHcStreams + i] * scale[2] + base[2 * kHcStreams + i];
    }
    for (int src = 0; src < kHcStreams; ++src) {
      double mx = -1e30;
      double sum = 0.0;
      for (int d = 0; d < kHcStreams; ++d) {
        mx = fmax(mx, static_cast<double>(comb[d + kHcStreams * src]));
      }
      for (int d = 0; d < kHcStreams; ++d) {
        comb[d + kHcStreams * src] =
            static_cast<float>(exp(comb[d + kHcStreams * src] - mx));
        sum += comb[d + kHcStreams * src];
      }
      for (int d = 0; d < kHcStreams; ++d) {
        comb[d + kHcStreams * src] /= static_cast<float>(sum + eps);
      }
    }
    for (int round = 0; round <= kSinkhornRounds * 2; ++round) {
      // First pass normalizes over src, then 19 x (over dst, over src).
      const bool over_src = round == 0 || (round % 2) == 0;
      for (int a = 0; a < kHcStreams; ++a) {
        double sum = 0.0;
        for (int b = 0; b < kHcStreams; ++b) {
          sum += over_src ? comb[a + kHcStreams * b] : comb[b + kHcStreams * a];
        }
        const float denom = static_cast<float>(sum + eps);
        for (int b = 0; b < kHcStreams; ++b) {
          if (over_src) {
            comb[a + kHcStreams * b] /= denom;
          } else {
            comb[b + kHcStreams * a] /= denom;
          }
        }
      }
    }
    for (int i = 0; i < kHcStreams * kHcStreams; ++i) {
      comb_out[i] = comb[i];
    }
  }
  __syncthreads();

  float z[kPerThread];
  float sq = 0.0F;
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    float v = 0.0F;
    for (int s = 0; s < kHcStreams; ++s) {
      v += x[s * kHcHidden + i] * pre[s];
    }
    z[k] = v;
    sq += v * v;
  }
  const float inv = rsqrtf(BlockSum(sq, scratch) / kHcHidden + kRmsEps);
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    y[i] = z[k] * inv * norm_w[i];
  }
}

__global__ void HcPostKernel(const float* __restrict__ x,
                             const float* __restrict__ block,
                             const float* __restrict__ post,
                             const float* __restrict__ comb,
                             float* __restrict__ next) {
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
    xs[s] = x[s * kHcHidden + i];
  }
  for (int d = 0; d < kHcStreams; ++d) {
    float acc = b * post[d];
    for (int s = 0; s < kHcStreams; ++s) {
      acc += xs[s] * comb[d + kHcStreams * s];
    }
    next[d * kHcHidden + i] = acc;
  }
}

__global__ __launch_bounds__(kThreads) void HeadNormKernel(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out) {
  __shared__ float scratch[kMaxWaves];
  const int tid = threadIdx.x;
  float v[kPerThread];
  float sq = 0.0F;
  for (int k = 0; k < kPerThread; ++k) {
    const int i = tid + k * kThreads;
    float acc = 0.0F;
    for (int s = 0; s < kHcStreams; ++s) {
      acc += x[s * kHcHidden + i];
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

}  // namespace

int HcPre(const float* x, const float* fn, const float* scale,
          const float* base, const float* norm_w, float hc_eps, float* y,
          float* post, float* comb, hipStream_t stream, int tokens) {
  HcPreKernel<<<tokens, kThreads, 0, stream>>>(x, fn, scale, base, norm_w,
                                               hc_eps, y, post, comb);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HcPost(const float* x, const float* block, const float* post,
           const float* comb, float* next, hipStream_t stream, int tokens) {
  constexpr int kBlock = 256;
  HcPostKernel<<<dim3((kHcHidden + kBlock - 1) / kBlock, tokens), kBlock, 0,
                 stream>>>(x, block, post, comb, next);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

int HeadNorm(const float* x, const float* w, float* out, hipStream_t stream) {
  HeadNormKernel<<<1, kThreads, 0, stream>>>(x, w, out);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}

}  // namespace gufo::models::glm53_flash::rocm
