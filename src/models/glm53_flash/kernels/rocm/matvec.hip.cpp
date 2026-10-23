#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

#include "ds4_mmq.h"
#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"

// The vendored mmq tier asks the DS4 backend for producer-folded activations.
// GLM never registers any; the weak definition yields to the real one when the
// DS4 backend is linked into the same binary.
extern "C" __attribute__((weak)) int ds4_hip_q8_fold_take_q81(
    const void*, std::uint64_t, const void** q81) {
  *q81 = nullptr;
  return 0;
}

namespace gufo::models::glm53_flash::rocm {

namespace {

constexpr int kF32Threads = 256;

__global__ void MatvecF32Kernel(const float* __restrict__ w,
                                const float* __restrict__ x,
                                float* __restrict__ out, int m, int n, int k) {
  const int row = blockIdx.x;
  const int col = blockIdx.y;
  const float* wr = w + static_cast<std::size_t>(row) * k;
  const float* xc = x + static_cast<std::size_t>(col) * k;
  float acc = 0.0F;
  for (int i = threadIdx.x; i < k; i += kF32Threads) {
    acc += wr[i] * xc[i];
  }
  __shared__ float partial[kF32Threads];
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kF32Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[static_cast<std::size_t>(col) * m + row] = partial[0];
  }
}

}  // namespace

std::size_t MatvecScratchBytes(int n, int k) {
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  return static_cast<std::size_t>(n) * padded * sizeof(block_q8_1) / QK8_1;
}

int Matvec(int type_code, const void* w, const float* x, float* out, int m,
           int n, int k, void* scratch, hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || m <= 0 || n <= 0 ||
      k <= 0) {
    return -1;
  }
  if (type_code == kTypeF32) {
    MatvecF32Kernel<<<dim3(m, n), kF32Threads, 0, stream>>>(
        static_cast<const float*>(w), x, out, m, n, k);
    return hipGetLastError() == hipSuccess ? 0 : -2;
  }
  if (scratch == nullptr || n > MMVQ_MAX_BATCH_SIZE) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * n,
      static_cast<std::int64_t>(k) * n, padded, n, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, nullptr, fusion, out, k, m, n,
                            stride_row_x, stride_col_y, m, 1, 1, 1, 0,
                            n * stride_col_y, 0, 1, 1, 0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecBatched(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int channels, void* scratch,
                  hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || scratch == nullptr ||
      type_code == kTypeF32 || m <= 0 || k <= 0 || channels <= 0) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(
      x, nullptr, scratch, type, k, k, static_cast<std::int64_t>(k) * channels,
      static_cast<std::int64_t>(k) * channels, padded, channels, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, nullptr, fusion, out, k, m, 1,
                            stride_row_x, stride_col_y, m, channels, channels,
                            channels, m * stride_row_x, stride_col_y, m, 1, 1,
                            0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int MatvecExperts(int type_code, const void* w, const float* x, float* out,
                  int m, int k, int n_experts, int n_used, const int* ids,
                  bool x_per_expert, void* scratch, hipStream_t stream) {
  if (w == nullptr || x == nullptr || out == nullptr || scratch == nullptr ||
      ids == nullptr || type_code == kTypeF32 || m <= 0 || k <= 0 ||
      n_used <= 0 || n_experts <= 0) {
    return -1;
  }
  const auto type = static_cast<ggml_type>(type_code);
  const int channels_y = x_per_expert ? n_used : 1;
  const std::int64_t padded =
      GGML_PAD(static_cast<std::int64_t>(k), MATRIX_ROW_PADDING);
  ds4_quantize_row_q8_1_hip(x, nullptr, scratch, type, k, k,
                            static_cast<std::int64_t>(k) * channels_y,
                            static_cast<std::int64_t>(k) * channels_y, padded,
                            channels_y, 1, 1, stream);
  const int blck = static_cast<int>(ggml_blck_size(type));
  const int stride_row_x = k / blck;
  const int stride_col_y = static_cast<int>(padded / QK8_1);
  ds4_ggml_hip_mm_fusion_args_device fusion = {};
  mul_mat_vec_q_switch_type(w, type, scratch, ids, fusion, out, k, m, 1,
                            stride_row_x, stride_col_y, m, n_experts,
                            channels_y, n_used, m * stride_row_x, stride_col_y,
                            m, 1, 1, 0, 0, 0, 0, stream);
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

int Gemm(int type_code, const void* w, const float* x, float* out, int m, int n,
         int k, void* scratch, hipStream_t stream) {
  if (type_code == kTypeF32 || n <= MMVQ_MAX_BATCH_SIZE) {
    return Matvec(type_code, w, x, out, m, n, k, scratch, stream);
  }
  if ((k % 256) != 0) {
    for (int col = 0; col < n; col += MMVQ_MAX_BATCH_SIZE) {
      const int cols =
          n - col < MMVQ_MAX_BATCH_SIZE ? n - col : MMVQ_MAX_BATCH_SIZE;
      const int rc = Matvec(type_code, w, x + static_cast<std::size_t>(col) * k,
                            out + static_cast<std::size_t>(col) * m, m, cols, k,
                            scratch, stream);
      if (rc != 0) {
        return rc;
      }
    }
    return 0;
  }
  switch (type_code) {
    case kTypeQ6_K:
      return ds4_mmq_q6_K_dense(w, x, out, m, n, k, stream);
    case kTypeQ8_0:
      return ds4_mmq_q8_0_dense(w, x, out, m, n, k, stream);
    default:
      return -1;
  }
}

int ExpertsGemm(int type_code, const void* w, const float* x, float* out, int m,
                int k, int tokens, int n_experts, int n_used, const int* ids,
                hipStream_t stream) {
  switch (type_code) {
    case kTypeIQ2_XXS:
      return ds4_mmq_iq2_xxs_moe(w, x, ids, out, m, k, tokens, n_experts,
                                 n_used, stream);
    case kTypeIQ2_S:
      return ds4_mmq_iq2_s_moe(w, x, ids, out, m, k, tokens, n_experts, n_used,
                               stream);
    default:
      return -1;
  }
}

}  // namespace gufo::models::glm53_flash::rocm
