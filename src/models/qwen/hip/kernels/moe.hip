#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/ops/moe.hpp"

namespace gufo::hip {

namespace {

constexpr std::uint32_t kMoeRouterThreads = 256;

__global__ void MoeRouterTopKKernel(const float* __restrict__ router_logits,
                                    const float* __restrict__ shared_logit,
                                    std::int32_t* __restrict__ expert_ids,
                                    float* __restrict__ expert_weights,
                                    float* __restrict__ shared_weight,
                                    std::uint32_t num_experts,
                                    std::uint32_t num_used) {
  // One block per token row.
  router_logits += static_cast<std::size_t>(blockIdx.x) * num_experts;
  shared_logit += blockIdx.x;
  expert_ids += static_cast<std::size_t>(blockIdx.x) * num_used;
  expert_weights += static_cast<std::size_t>(blockIdx.x) * num_used;
  shared_weight += blockIdx.x;
  extern __shared__ float probs[];
  __shared__ float s_max[8];
  __shared__ float s_sum[8];

  const std::uint32_t tid = threadIdx.x;

  float local_max = -INFINITY;
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, router_logits[e]);
  }
  float warp_max = local_max;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_max = fmaxf(warp_max, __shfl_xor(warp_max, offset));
  }
  const std::uint32_t warp = tid >> 5;
  const std::uint32_t lane = tid & 31U;
  if (lane == 0) {
    s_max[warp] = warp_max;
  }
  __syncthreads();
  const std::uint32_t warp_count = blockDim.x >> 5;
  if (tid == 0) {
    float block_max = s_max[0];
    for (std::uint32_t w = 1; w < warp_count; ++w) {
      block_max = fmaxf(block_max, s_max[w]);
    }
    s_max[0] = block_max;
  }
  __syncthreads();
  const float peak = s_max[0];

  float local_sum = 0.0F;
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    const float v = expf(router_logits[e] - peak);
    probs[e] = v;
    local_sum += v;
  }
  float warp_sum = local_sum;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_sum += __shfl_xor(warp_sum, offset);
  }
  if (lane == 0) {
    s_sum[warp] = warp_sum;
  }
  __syncthreads();
  if (tid == 0) {
    float block_sum = 0.0F;
    for (std::uint32_t w = 0; w < warp_count; ++w) {
      block_sum += s_sum[w];
    }
    s_sum[0] = block_sum;
  }
  __syncthreads();
  const float total = s_sum[0];

  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    probs[e] /= total;
  }
  __syncthreads();

  // Top-k by repeated block-wide argmax: ties resolve toward the lower index,
  // as a stable sort would.
  __shared__ float s_best_value[8];
  __shared__ std::int32_t s_best_index[8];
  __shared__ float s_pick_value[64];
  __shared__ std::int32_t s_pick_index[64];
  const std::uint32_t capacity =
      num_used < 64 ? num_used : static_cast<std::uint32_t>(64);
  for (std::uint32_t k = 0; k < capacity; ++k) {
    float best = -1.0F;
    std::int32_t best_e = 0x7fffffff;
    for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
      const float v = probs[e];
      if (v > best || (v == best && static_cast<std::int32_t>(e) < best_e)) {
        best = v;
        best_e = static_cast<std::int32_t>(e);
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_xor(best, offset);
      const std::int32_t other_e = __shfl_xor(best_e, offset);
      if (other > best || (other == best && other_e < best_e)) {
        best = other;
        best_e = other_e;
      }
    }
    if (lane == 0) {
      s_best_value[warp] = best;
      s_best_index[warp] = best_e;
    }
    __syncthreads();
    if (tid == 0) {
      float block_best = s_best_value[0];
      std::int32_t block_e = s_best_index[0];
      for (std::uint32_t w = 1; w < warp_count; ++w) {
        if (s_best_value[w] > block_best ||
            (s_best_value[w] == block_best && s_best_index[w] < block_e)) {
          block_best = s_best_value[w];
          block_e = s_best_index[w];
        }
      }
      s_pick_value[k] = block_best;
      s_pick_index[k] = block_e;
      probs[block_e] = -1.0F;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float selected_sum = 0.0F;
    for (std::uint32_t k = 0; k < capacity; ++k) {
      selected_sum += s_pick_value[k];
    }
    const float inv = selected_sum > 0.0F ? 1.0F / selected_sum : 0.0F;
    for (std::uint32_t k = 0; k < capacity; ++k) {
      expert_ids[k] = s_pick_index[k];
      expert_weights[k] = s_pick_value[k] * inv;
    }
    shared_weight[0] = 1.0F / (1.0F + expf(-shared_logit[0]));
  }
}

__global__ void MoeScaledAccumulateKernel(float* __restrict__ out,
                                          const float* __restrict__ add,
                                          float scale, std::size_t dim) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    out[idx] += scale * add[idx];
  }
}

__global__ void MoeCombineKernel(float* __restrict__ out,
                                 const float* __restrict__ slot_out,
                                 const float* __restrict__ slot_weight,
                                 const float* __restrict__ shared_out,
                                 const float* __restrict__ shared_weight,
                                 std::uint32_t used, std::uint32_t hidden) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) {
    return;
  }
  const std::size_t row = blockIdx.y;
  out += row * hidden;
  slot_out += row * used * hidden;
  slot_weight += row * used;
  shared_out += row * hidden;
  shared_weight += row;
  float sum = 0.0F;
  for (std::uint32_t s = 0; s < used; ++s) {
    sum += slot_weight[s] * slot_out[static_cast<std::size_t>(s) * hidden + i];
  }
  out[i] = sum + shared_weight[0] * shared_out[i];
}

__device__ inline std::uint16_t FloatToBf16Bits(float f) {
  std::uint32_t u = __float_as_uint(f);
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<std::uint16_t>(u >> 16);
}

__device__ inline float Bf16BitsToFloat(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

__global__ void RouterSplitActivationsKernel(const float* __restrict__ x,
                                             std::uint16_t* __restrict__ dst,
                                             std::size_t k) {
  const float* row = x + (static_cast<std::size_t>(blockIdx.x) * k);
  std::uint16_t* out = dst + (static_cast<std::size_t>(blockIdx.x) * 3 * k);
  for (std::size_t i = threadIdx.x; i < k; i += blockDim.x) {
    const std::uint16_t hi = FloatToBf16Bits(row[i]);
    const std::uint16_t lo = FloatToBf16Bits(row[i] - Bf16BitsToFloat(hi));
    out[i] = hi;
    out[k + i] = lo;
    out[(2 * k) + i] = hi;
  }
}

__global__ void RouterSplitWeightsKernel(const void* __restrict__ src,
                                         bool src_bf16,
                                         std::uint16_t* __restrict__ dst,
                                         std::size_t first_row, std::size_t k) {
  const std::size_t r = blockIdx.x;
  std::uint16_t* out = dst + ((first_row + r) * 3 * k);
  for (std::size_t i = threadIdx.x; i < k; i += blockDim.x) {
    std::uint16_t hi;
    std::uint16_t lo = 0;
    if (src_bf16) {
      hi = static_cast<const std::uint16_t*>(src)[(r * k) + i];
    } else {
      const float w = static_cast<const float*>(src)[(r * k) + i];
      hi = FloatToBf16Bits(w);
      lo = FloatToBf16Bits(w - Bf16BitsToFloat(hi));
    }
    out[i] = hi;
    out[k + i] = hi;
    out[(2 * k) + i] = lo;
  }
}

__global__ void RouterSplitLogitsKernel(const float* __restrict__ y,
                                        float* __restrict__ router,
                                        float* __restrict__ shared,
                                        std::size_t experts, std::size_t ld) {
  const float* row = y + (static_cast<std::size_t>(blockIdx.x) * ld);
  for (std::size_t i = threadIdx.x; i < experts; i += blockDim.x) {
    router[(static_cast<std::size_t>(blockIdx.x) * experts) + i] = row[i];
  }
  if (threadIdx.x == 0) {
    shared[blockIdx.x] = row[experts];
  }
}

}  // namespace

void LaunchRouterSplitActivations(const float* x, std::uint16_t* dst,
                                  std::size_t rows, std::size_t k,
                                  hipStream_t stream) {
  RouterSplitActivationsKernel<<<dim3(rows), 256, 0, stream>>>(x, dst, k);
}

void LaunchRouterSplitWeights(const void* src, bool src_bf16,
                              std::uint16_t* dst, std::size_t first_row,
                              std::size_t rows, std::size_t k,
                              hipStream_t stream) {
  RouterSplitWeightsKernel<<<dim3(rows), 256, 0, stream>>>(src, src_bf16, dst,
                                                           first_row, k);
}

void LaunchRouterSplitLogits(const float* y, float* router, float* shared,
                             std::size_t rows, std::size_t experts,
                             std::size_t ld, hipStream_t stream) {
  RouterSplitLogitsKernel<<<dim3(rows), 256, 0, stream>>>(y, router, shared,
                                                          experts, ld);
}

void LaunchMoeRouterTopK(const float* router_logits, const float* shared_logit,
                         std::int32_t* expert_ids, float* expert_weights,
                         float* shared_weight, std::uint32_t num_experts,
                         std::uint32_t num_used, std::uint32_t rows,
                         hipStream_t stream) {
  const std::size_t shared_bytes =
      static_cast<std::size_t>(num_experts) * sizeof(float);
  MoeRouterTopKKernel<<<rows, kMoeRouterThreads, shared_bytes, stream>>>(
      router_logits, shared_logit, expert_ids, expert_weights, shared_weight,
      num_experts, num_used);
}

void LaunchMoeScaledAccumulate(float* out, const float* add, float scale,
                               std::size_t dim, hipStream_t stream) {
  constexpr std::size_t kBlockSize = 256;
  const std::size_t grid = (dim + kBlockSize - 1) / kBlockSize;
  MoeScaledAccumulateKernel<<<grid, kBlockSize, 0, stream>>>(out, add, scale,
                                                             dim);
}

void LaunchMoeCombine(float* out, const float* slot_out,
                      const float* slot_weight, const float* shared_out,
                      const float* shared_weight, std::uint32_t used,
                      std::uint32_t hidden, std::uint32_t rows,
                      hipStream_t stream) {
  MoeCombineKernel<<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(
      out, slot_out, slot_weight, shared_out, shared_weight, used, hidden);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
