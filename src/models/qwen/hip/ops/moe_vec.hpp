#ifndef GUFO_MODELS_QWEN_HIP_OPS_MOE_VEC_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_MOE_VEC_HPP_

#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

// Expert vector kernels of the quantized matrix library (see
// src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp).
extern "C" {
/// out[t][s][m] = SiLU(gate_e . x_t) * (up_e . x_t) with e = ids[t][s].
int qfn_mmq_moe_gated_vec(int weight_type, const void* gate, const void* up,
                          const float* x, const std::int32_t* ids, float* out,
                          int m, int k, int tokens, int experts,
                          int experts_used, hipStream_t stream);
/// out[t][s][m] = W_e . x_t (x_t per row) with e = ids[t][s].
int qfn_mmq_moe_vec(int weight_type, const void* w, const float* x,
                    const std::int32_t* ids, float* out, int m, int k,
                    int n_tokens, int n_experts, int n_expert_used,
                    hipStream_t stream, const void* w_b, float* out_b);
}

#endif  // defined(ENGINE_ENABLE_HIP)
#endif  // GUFO_MODELS_QWEN_HIP_OPS_MOE_VEC_HPP_
