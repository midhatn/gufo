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

  if (tid == 0) {
    float best_val[64];
    std::int32_t best_idx[64];
    const std::uint32_t capacity =
        num_used < 64 ? num_used : static_cast<std::uint32_t>(64);
    std::uint32_t n = 0;
    for (std::uint32_t e = 0; e < num_experts; ++e) {
      const float v = probs[e];
      std::uint32_t pos = n;
      for (std::uint32_t k = 0; k < n; ++k) {
        if (v > best_val[k]) {
          pos = k;
          break;
        }
      }
      if (pos < capacity) {
        const std::uint32_t last = n < capacity ? n : capacity - 1;
        for (std::uint32_t k = last; k > pos; --k) {
          best_val[k] = best_val[k - 1];
          best_idx[k] = best_idx[k - 1];
        }
        best_val[pos] = v;
        best_idx[pos] = static_cast<std::int32_t>(e);
        if (n < capacity)
          ++n;
      }
    }
    float selected_sum = 0.0F;
    for (std::uint32_t k = 0; k < n; ++k) {
      selected_sum += best_val[k];
    }
    const float inv = selected_sum > 0.0F ? 1.0F / selected_sum : 0.0F;
    for (std::uint32_t k = 0; k < capacity; ++k) {
      if (k < n) {
        expert_ids[k] = best_idx[k];
        expert_weights[k] = best_val[k] * inv;
      } else {
        expert_ids[k] = 0;
        expert_weights[k] = 0.0F;
      }
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

}  // namespace

void LaunchMoeRouterTopK(const float* router_logits, const float* shared_logit,
                         std::int32_t* expert_ids, float* expert_weights,
                         float* shared_weight, std::uint32_t num_experts,
                         std::uint32_t num_used, hipStream_t stream) {
  const std::size_t shared_bytes =
      static_cast<std::size_t>(num_experts) * sizeof(float);
  MoeRouterTopKKernel<<<1, kMoeRouterThreads, shared_bytes, stream>>>(
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

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
