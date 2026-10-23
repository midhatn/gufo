#include "src/models/glm53_flash/kernels/rocm/matvec.hpp"

#include "mmvq.hip.hpp"
#include "quantize.hip.hpp"

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
      k <= 0 || n > MMVQ_MAX_BATCH_SIZE) {
    return -1;
  }
  if (type_code == kTypeF32) {
    MatvecF32Kernel<<<dim3(m, n), kF32Threads, 0, stream>>>(
        static_cast<const float*>(w), x, out, m, n, k);
    return hipGetLastError() == hipSuccess ? 0 : -2;
  }
  if (scratch == nullptr) {
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

}  // namespace gufo::models::glm53_flash::rocm
