// Chunked (WY form) gated delta rule for the Flash-Next GDN layers on gfx1151,
// against the exact sequential recurrence.
//
// Build: nix develop -c tools/bench/build.sh gdn_chunk_bench
// Run:   /tmp/gdn_chunk_bench [tokens]
//
// Per head the recurrence is S <- g S (I - b k k^T) + b v k^T, o = S q. With
// chunks of 64 tokens the state-independent part (K K^T, Q K^T and the inverse
// of the unit lower-triangular system) is computed for all chunks in parallel;
// the sequential kernel then only runs matrix products per chunk.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define HIP_OK(expr)                                                      \
  do {                                                                    \
    const hipError_t err_ = (expr);                                       \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__,   \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

constexpr int kD = 128;
constexpr int kKH = 16;
constexpr int kH = 48;
constexpr int kC = 64;
constexpr int kChannels = 2 * kKH * kD + kH * kD;  // 10240
constexpr float kEps = 1e-6f;

// ---------------------------------------------------------------------------
// Reference: one thread per state row, exactly the production arithmetic
// (raw keys, scales from the K/Q pre-pass).
__global__ void RefKernel(const __half* conv, const float* scales,
                          const float* ab, float* state, float* raw,
                          int n_tokens) {
  const int h = blockIdx.x;
  const int j = blockIdx.y * blockDim.x + threadIdx.x;
  const int kh = h % kKH;
  float* S = state + (static_cast<size_t>(h) * kD + j) * kD;
  float row[kD];
  for (int i = 0; i < kD; ++i) row[i] = S[i];
  for (int t = 0; t < n_tokens; ++t) {
    const __half* q = conv + static_cast<size_t>(t) * kChannels + kh * kD;
    const __half* k = conv + static_cast<size_t>(t) * kChannels + (kKH + kh) * kD;
    const float vv = __half2float(
        conv[static_cast<size_t>(t) * kChannels + 2 * kKH * kD + h * kD + j]);
    const float inv_k = scales[(static_cast<size_t>(t) * kKH + kh) * 3 + 0];
    const float q_scale = scales[(static_cast<size_t>(t) * kKH + kh) * 3 + 1];
    const float kq = scales[(static_cast<size_t>(t) * kKH + kh) * 3 + 2];
    const float decay = ab[(static_cast<size_t>(t) * kH + h) * 2];
    const float beta = ab[(static_cast<size_t>(t) * kH + h) * 2 + 1];
    float u = 0.f, p = 0.f;
    for (int i = 0; i < kD; ++i) {
      row[i] *= decay;
      u += row[i] * __half2float(k[i]);
      p += row[i] * __half2float(q[i]);
    }
    const float delta = (vv - u * inv_k) * beta;
    const float update = delta * inv_k;
    raw[static_cast<size_t>(t) * kH * kD + h * kD + j] = (p + kq * update) * q_scale;
    for (int i = 0; i < kD; ++i) row[i] += update * __half2float(k[i]);
  }
  for (int i = 0; i < kD; ++i) S[i] = row[i];
}

// K/Q pre-pass shared by both paths: inv_k, q_scale, kq per (token, k head).
__global__ void ScalesKernel(const __half* conv, float* scales, int n_tokens) {
  const int t = blockIdx.x;
  const int kh = blockIdx.y;
  const int lane = threadIdx.x;  // 32
  float qs = 0, ks = 0, kq = 0;
  for (int i = lane; i < kD; i += 32) {
    const float q = __half2float(conv[static_cast<size_t>(t) * kChannels + kh * kD + i]);
    const float k = __half2float(conv[static_cast<size_t>(t) * kChannels + (kKH + kh) * kD + i]);
    qs += q * q; ks += k * k; kq += k * q;
  }
  for (int o = 16; o > 0; o >>= 1) {
    qs += __shfl_xor(qs, o); ks += __shfl_xor(ks, o); kq += __shfl_xor(kq, o);
  }
  if (lane == 0) {
    float* d = scales + (static_cast<size_t>(t) * kKH + kh) * 3;
    d[0] = rsqrtf(ks + kEps);
    d[1] = rsqrtf(qs + kEps) * rsqrtf(static_cast<float>(kD));
    d[2] = kq;
  }
}

// ---------------------------------------------------------------------------
// P0: normalised K and Q chunks as F16: Khat [chunk][kh][c][i],
// KhatT [chunk][kh][i][c], Qhat [chunk][kh][c][i].
__global__ __launch_bounds__(256) void NormKqKernel(
    const __half* __restrict__ conv, const float* __restrict__ scales,
    __half* __restrict__ khat, __half* __restrict__ khat_t,
    __half* __restrict__ qhat, __half* __restrict__ kkmat,
    __half* __restrict__ qkmat) {
  __shared__ __attribute__((aligned(16))) __half tile[kC][kD + 8];
  __shared__ __attribute__((aligned(16))) __half qtile[kC][kD + 8];
  const int kh = blockIdx.x;
  const int chunk = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const size_t base = (static_cast<size_t>(chunk) * kKH + kh);
  for (int idx = tid; idx < kC * (kD / 8); idx += 256) {
    const int c = idx / (kD / 8), i8 = (idx % (kD / 8)) * 8;
    const int t = chunk * kC + c;
    const float* sc = scales + (static_cast<size_t>(t) * kKH + kh) * 3;
    const float sk = sc[0], sq = sc[1];
    const uint4 kraw = *reinterpret_cast<const uint4*>(
        conv + static_cast<size_t>(t) * kChannels + (kKH + kh) * kD + i8);
    const uint4 qraw = *reinterpret_cast<const uint4*>(
        conv + static_cast<size_t>(t) * kChannels + kh * kD + i8);
    const __half* kp = reinterpret_cast<const __half*>(&kraw);
    const __half* qp = reinterpret_cast<const __half*>(&qraw);
    __half ko[8], qo[8];
    for (int e = 0; e < 8; ++e) {
      ko[e] = __float2half_rn(__half2float(kp[e]) * sk);
      qo[e] = __float2half_rn(__half2float(qp[e]) * sq);
    }
    *reinterpret_cast<uint4*>(&tile[c][i8]) = *reinterpret_cast<const uint4*>(ko);
    *reinterpret_cast<uint4*>(&qtile[c][i8]) = *reinterpret_cast<const uint4*>(qo);
    *reinterpret_cast<uint4*>(khat + base * kC * kD + c * kD + i8) = *reinterpret_cast<const uint4*>(ko);
    *reinterpret_cast<uint4*>(qhat + base * kC * kD + c * kD + i8) = *reinterpret_cast<const uint4*>(qo);
  }
  __syncthreads();
  // K^T rows for the state update.
  for (int idx = tid; idx < kD * (kC / 8); idx += 256) {
    const int i = idx % kD, c8 = (idx / kD) * 8;
    __half out[8];
    for (int e = 0; e < 8; ++e) out[e] = tile[c8 + e][i];
    *reinterpret_cast<uint4*>(khat_t + base * kC * kD + i * kC + c8) =
        *reinterpret_cast<const uint4*>(out);
  }
  // Raw products KK = K K^T and QK = Q K^T (F16), 16 tiles over 8 waves.
  for (int tl = wave; tl < 16; tl += 8) {
    const int ta = tl / 4, tb = tl % 4;
    v8f kk = {}, qk = {};
#pragma unroll
    for (int it = 0; it < kD / 16; ++it) {
      v16h fa = *reinterpret_cast<const v16h*>(&tile[ta * 16 + (lane & 15)][it * 16]);
      v16h fq = *reinterpret_cast<const v16h*>(&qtile[ta * 16 + (lane & 15)][it * 16]);
      v16h fb = *reinterpret_cast<const v16h*>(&tile[tb * 16 + (lane & 15)][it * 16]);
      kk = Wmma(fa, fb, kk);
      qk = Wmma(fq, fb, qk);
    }
    for (int r = 0; r < 8; ++r) {
      const int a = ta * 16 + 2 * r + (lane >> 4);
      const int b = tb * 16 + (lane & 15);
      kkmat[(base * kC + a) * kC + b] = __float2half_rn(kk[r]);
      qkmat[(base * kC + a) * kC + b] = __float2half_rn(qk[r]);
    }
  }
}

// ---------------------------------------------------------------------------
// P1: per (chunk, head): cumulative decay, M = tril(Q K^T * D), and
// X = (I + A)^-1 diag(beta), A = tril(beta_i K K^T D, -1).
__global__ __launch_bounds__(256) void ChunkPrepKernel(
    const __half* __restrict__ kkmat, const __half* __restrict__ qkmat,
    const float* __restrict__ ab, __half* __restrict__ xmat,
    __half* __restrict__ mmat, float* __restrict__ gvec,
    float* __restrict__ wvec) {
  __shared__ float s_a[kC][kC + 1];
  __shared__ float s_y[kC][kC + 1];
  __shared__ float s_lg[kC];
  __shared__ float s_beta[kC];
  const int h = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kh = h % kKH;
  const int tid = threadIdx.x;
  const size_t kq_base = (static_cast<size_t>(chunk) * kKH + kh) * kC * kC;
  const size_t out_base = (static_cast<size_t>(chunk) * kH + h);

  if (tid < kC) {
    const int t = chunk * kC + tid;
    const float* p = ab + (static_cast<size_t>(t) * kH + h) * 2;
    s_lg[tid] = logf(fmaxf(p[0], 1e-30f));
    s_beta[tid] = p[1];
  }
  __syncthreads();
  if (tid < 32) {  // inclusive scan of the 64 log decays by one wave
    float v0 = s_lg[tid], v1 = s_lg[tid + 32];
    for (int o = 1; o < 32; o <<= 1) {
      const float n0 = __shfl_up(v0, o), n1 = __shfl_up(v1, o);
      if (tid >= o) { v0 += n0; v1 += n1; }
    }
    const float tail = __shfl(v0, 31);
    s_lg[tid] = v0;
    s_lg[tid + 32] = v1 + tail;
  }
  __syncthreads();
  if (tid < kC) {
    gvec[out_base * kC + tid] = expf(s_lg[tid]);
    wvec[out_base * kC + tid] = expf(s_lg[kC - 1] - s_lg[tid]);
  }
  // Decay-weighted A (strictly lower) and M (lower) from the raw products,
  // eight consecutive columns per thread.
  for (int idx = tid; idx < kC * (kC / 8); idx += 256) {
    const int a = idx / (kC / 8), b8 = (idx % (kC / 8)) * 8;
    const uint4 kk4 = *reinterpret_cast<const uint4*>(kkmat + kq_base + a * kC + b8);
    const uint4 qk4 = *reinterpret_cast<const uint4*>(qkmat + kq_base + a * kC + b8);
    const __half* kk = reinterpret_cast<const __half*>(&kk4);
    const __half* qk = reinterpret_cast<const __half*>(&qk4);
    __half mo[8];
    for (int e = 0; e < 8; ++e) {
      const int b = b8 + e;
      const float dij = b <= a ? expf(s_lg[a] - s_lg[b]) : 0.f;
      s_a[a][b] = b < a ? s_beta[a] * __half2float(kk[e]) * dij : 0.f;
      mo[e] = __float2half_rn(__half2float(qk[e]) * dij);
    }
    *reinterpret_cast<uint4*>(mmat + (out_base * kC + a) * kC + b8) = *reinterpret_cast<const uint4*>(mo);
  }
  __syncthreads();
  // Y = (I + A)^-1 by 16x16 blocks. L = I + A lives in s_a (identity added),
  // Y in s_y (zero above the diagonal blocks).
  for (int idx = tid; idx < kC * kC; idx += 256) {
    const int i = idx / kC, c = idx % kC;
    if (i == c) s_a[i][c] += 1.f;
    s_y[i][c] = 0.f;
  }
  __syncthreads();
  if (tid < kC) {  // diagonal blocks: thread = column within block
    const int b = tid / 16, cc = tid % 16, o = b * 16;
    float y[16];
    for (int i = 0; i < 16; ++i) {
      float acc = i == cc ? 1.f : 0.f;
      if (i > cc) {
        float sum = 0.f;
        for (int m = cc; m < i; ++m) sum += s_a[o + i][o + m] * y[m];
        acc = -sum;
      } else if (i < cc) {
        acc = 0.f;
      }
      y[i] = acc;
      s_y[o + i][o + cc] = acc;
    }
  }
  __syncthreads();
  __shared__ float s_z[3][16][16];
  for (int d = 1; d < 4; ++d) {
    const int pairs = 4 - d;
    // Z_ij = sum_{m=j}^{i-1} L_im Y_mj, one element per thread and pair.
    for (int e = tid; e < pairs * 256; e += 256) {
      const int pr = e / 256, r = (e / 16) % 16, c = e % 16;
      const int j = pr, i = pr + d;
      float sum = 0.f;
      for (int m = j; m < i; ++m)
        for (int k = 0; k < 16; ++k)
          sum += s_a[i * 16 + r][m * 16 + k] * s_y[m * 16 + k][j * 16 + c];
      s_z[pr][r][c] = sum;
    }
    __syncthreads();
    for (int e = tid; e < pairs * 256; e += 256) {
      const int pr = e / 256, r = (e / 16) % 16, c = e % 16;
      const int j = pr, i = pr + d;
      float sum = 0.f;
      for (int k = 0; k < 16; ++k)
        sum += s_y[i * 16 + r][i * 16 + k] * s_z[pr][k][c];
      s_y[i * 16 + r][j * 16 + c] = -sum;
    }
    __syncthreads();
  }
  for (int idx = tid; idx < kC * kC; idx += 256) {
    const int i = idx / kC, c = idx % kC;
    xmat[(out_base * kC + i) * kC + c] = __float2half_rn(s_y[i][c] * s_beta[c]);
  }
}

// ---------------------------------------------------------------------------
// P2: sequential over chunks. Block (head, row group of JG value rows).
template<int JG>
__global__ __launch_bounds__(256) void ChunkScanKernel(
    const __half* __restrict__ conv, const __half* __restrict__ khat,
    const __half* __restrict__ khat_t, const __half* __restrict__ qhat,
    const __half* __restrict__ xmat, const __half* __restrict__ mmat,
    const float* __restrict__ gvec, const float* __restrict__ wvec,
    float* __restrict__ state, __half* __restrict__ raw, int n_chunks,
    unsigned long long* prof = nullptr) {
  static_assert(JG == 32 || JG == 64 || JG == 128);
  constexpr int kJT = JG / 16;           // row tiles
  constexpr int kJTH = kJT / 2;          // row tiles per wave in steps 1-4
  constexpr int kCT = kC / 16;           // token tiles (4)
  constexpr int kIT = kD / 16;           // key tiles (8)
  // Region A: S (F16 [j][i]) during step 1, then U^T [j][c].
  // Region B: rhs^T during step 3, then the F16 output stage.
  constexpr int kASt = kD + 8, kBSt = kC + 8, kOSt = JG + 8;
  constexpr int kABytes = JG * kASt * 2 > JG * kBSt * 2 ? JG * kASt * 2 : JG * kBSt * 2;
  constexpr int kBBytes = JG * kBSt * 2 > kC * kOSt * 2 ? JG * kBSt * 2 : kC * kOSt * 2;
  __shared__ __attribute__((aligned(16))) std::uint8_t s_region_a[kABytes];
  __shared__ __attribute__((aligned(16))) std::uint8_t s_region_b[kBBytes];
  __shared__ float s_w[kC];
  auto* s_s = reinterpret_cast<__half*>(s_region_a);   // [JG][kASt]
  auto* s_ut = reinterpret_cast<__half*>(s_region_a);  // [JG][kBSt]
  auto* s_rt = reinterpret_cast<__half*>(s_region_b);  // [JG][kBSt]
  auto* s_os = reinterpret_cast<__half*>(s_region_b);  // [kC][kOSt]
  const int h = blockIdx.x;
  const int j0 = blockIdx.y * JG;
  const int kh = h % kKH;
  const int tid = threadIdx.x;
  const int wave = tid >> 5, lane = tid & 31;
  const int l16 = lane & 15, hf = lane >> 4;
  const int ct = wave & 3;                // token tile of steps 1-4
  const int jbase = (wave >> 2) * kJTH;   // first row tile of steps 1-4
  const int c_tok = ct * 16 + l16;        // this lane's token within the chunk

  v8f sacc[kJT];
  for (int jt = 0; jt < kJT; ++jt) {
    for (int r = 0; r < 8; ++r) {
      const int j = jt * 16 + 2 * r + hf;
      sacc[jt][r] = state[(static_cast<size_t>(h) * kD + j0 + j) * kD + wave * 16 + l16];
    }
  }
  unsigned long long tmark = clock64();
  const bool do_prof = prof != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && wave == 0 && lane == 0;
#define MARK(slot) do { if (do_prof) { unsigned long long n_ = clock64(); prof[slot] += n_ - tmark; tmark = n_; } } while (0)

  for (int chunk = 0; chunk < n_chunks; ++chunk) {
    const size_t kq_base = (static_cast<size_t>(chunk) * kKH + kh) * kC * kD;
    const size_t xm_base = (static_cast<size_t>(chunk) * kH + h) * kC * kC;
    const size_t g_base = (static_cast<size_t>(chunk) * kH + h) * kC;
    const size_t trow = static_cast<size_t>(chunk) * kC + c_tok;
    // Operands independent of the state, requested before anything waits.
    v16h fx[kCT], fm[kCT];
#pragma unroll
    for (int cs = 0; cs < kCT; ++cs) {
      if (cs <= ct) {
        fx[cs] = *reinterpret_cast<const v16h*>(xmat + xm_base + c_tok * kC + cs * 16);
        fm[cs] = *reinterpret_cast<const v16h*>(mmat + xm_base + c_tok * kC + cs * 16);
      }
    }
    __half vv[kJTH][8];
#pragma unroll
    for (int tj = 0; tj < kJTH; ++tj)
      for (int r = 0; r < 8; ++r)
        vv[tj][r] = conv[trow * kChannels + 2 * kKH * kD + h * kD + j0 + (jbase + tj) * 16 + 2 * r + hf];
    const float g = gvec[g_base + c_tok];
    if (tid < kC) s_w[tid] = wvec[g_base + tid];
    for (int jt = 0; jt < kJT; ++jt) {
      for (int r = 0; r < 8; ++r) {
        s_s[(jt * 16 + 2 * r + hf) * kASt + wave * 16 + l16] = __float2half_rn(sacc[jt][r]);
      }
    }
    __syncthreads();
    MARK(0);
    if (chunk > 0) {
      for (int idx = tid; idx < kC * (JG / 8); idx += 256) {
        const int c = idx / (JG / 8), j8 = (idx % (JG / 8)) * 8;
        *reinterpret_cast<uint4*>(raw + (static_cast<size_t>(chunk - 1) * kC + c) * kH * kD + h * kD + j0 + j8) =
            *reinterpret_cast<const uint4*>(&s_os[c * kOSt + j8]);
      }
    }

    // Step 1: P = S K^T, Q' = S Q^T for this wave's token tile and row tiles.
    v8f pt[kJTH], qa[kJTH];
    for (int tj = 0; tj < kJTH; ++tj) { pt[tj] = v8f{}; qa[tj] = v8f{}; }
#pragma unroll
    for (int it = 0; it < kIT; ++it) {
      v16h fk = *reinterpret_cast<const v16h*>(khat + kq_base + c_tok * kD + it * 16);
      v16h fq = *reinterpret_cast<const v16h*>(qhat + kq_base + c_tok * kD + it * 16);
#pragma unroll
      for (int tj = 0; tj < kJTH; ++tj) {
        v16h fs = *reinterpret_cast<const v16h*>(&s_s[((jbase + tj) * 16 + l16) * kASt + it * 16]);
        pt[tj] = Wmma(fs, fk, pt[tj]);
        qa[tj] = Wmma(fs, fq, qa[tj]);
      }
    }
    MARK(1);
    __syncthreads();  // S is dead from here: region A is reused by U^T.
    for (int tj = 0; tj < kJTH; ++tj) {
      for (int r = 0; r < 8; ++r) {
        const int j = (jbase + tj) * 16 + 2 * r + hf;
        qa[tj][r] *= g;
        s_rt[j * kBSt + c_tok] = __float2half_rn(__half2float(vv[tj][r]) - g * pt[tj][r]);
      }
    }
    MARK(2);
    __syncthreads();
    MARK(3);
    // Step 3: U^T = rhs^T X^T.
    v8f u[kJTH];
    for (int tj = 0; tj < kJTH; ++tj) u[tj] = v8f{};
#pragma unroll
    for (int cs = 0; cs < kCT; ++cs) {
      if (cs <= ct) {
#pragma unroll
        for (int tj = 0; tj < kJTH; ++tj) {
          v16h fa = *reinterpret_cast<const v16h*>(&s_rt[((jbase + tj) * 16 + l16) * kBSt + cs * 16]);
          u[tj] = Wmma(fa, fx[cs], u[tj]);
        }
      }
    }
    for (int tj = 0; tj < kJTH; ++tj) {
      for (int r = 0; r < 8; ++r) {
        s_ut[((jbase + tj) * 16 + 2 * r + hf) * kBSt + c_tok] = __float2half_rn(u[tj][r]);
      }
    }
    MARK(4);
    __syncthreads();
    MARK(5);
    // Step 4: O^T = G Q' + U^T M^T. The output stage overwrites rhs^T (dead).
#pragma unroll
    for (int cs = 0; cs < kCT; ++cs) {
      if (cs <= ct) {
#pragma unroll
        for (int tj = 0; tj < kJTH; ++tj) {
          v16h fa = *reinterpret_cast<const v16h*>(&s_ut[((jbase + tj) * 16 + l16) * kBSt + cs * 16]);
          qa[tj] = Wmma(fa, fm[cs], qa[tj]);
        }
      }
    }
    for (int tj = 0; tj < kJTH; ++tj) {
      for (int r = 0; r < 8; ++r) {
        s_os[c_tok * kOSt + (jbase + tj) * 16 + 2 * r + hf] = __float2half_rn(qa[tj][r]);
      }
    }
    MARK(6);
    // Step 5: S = Gc S + (w U)^T K, wave w owns key tile w and every row tile.
    const float gc = gvec[g_base + kC - 1];
    v16h fkt[kC / 16];
#pragma unroll
    for (int cs = 0; cs < kC / 16; ++cs) {
      fkt[cs] = *reinterpret_cast<const v16h*>(khat_t + kq_base + (wave * 16 + l16) * kC + cs * 16);
#pragma unroll
      for (int e = 0; e < 16; ++e)
        fkt[cs][e] = static_cast<_Float16>(static_cast<float>(fkt[cs][e]) * s_w[cs * 16 + e]);
    }
#pragma unroll
    for (int jt = 0; jt < kJT; ++jt) {
      for (int r = 0; r < 8; ++r) sacc[jt][r] *= gc;
#pragma unroll
      for (int cs = 0; cs < kC / 16; ++cs) {
        v16h fa = *reinterpret_cast<const v16h*>(&s_ut[(jt * 16 + l16) * kBSt + cs * 16]);
        sacc[jt] = Wmma(fa, fkt[cs], sacc[jt]);
      }
    }
    MARK(7);
    __syncthreads();
    MARK(8);
  }
  for (int idx = tid; idx < kC * (JG / 8); idx += 256) {
    const int c = idx / (JG / 8), j8 = (idx % (JG / 8)) * 8;
    *reinterpret_cast<uint4*>(raw + (static_cast<size_t>(n_chunks - 1) * kC + c) * kH * kD + h * kD + j0 + j8) =
        *reinterpret_cast<const uint4*>(&s_os[c * kOSt + j8]);
  }
  for (int jt = 0; jt < kJT; ++jt) {
    for (int r = 0; r < 8; ++r) {
      const int j = jt * 16 + 2 * r + hf;
      state[(static_cast<size_t>(h) * kD + j0 + j) * kD + wave * 16 + l16] = sacc[jt][r];
    }
  }
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 8192;
  const int n_chunks = T / kC;
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  std::vector<__half> conv(static_cast<size_t>(T) * kChannels);
  for (auto& x : conv) x = __float2half_rn(nd(rng));
  std::vector<float> ab(static_cast<size_t>(T) * kH * 2);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < kH; ++h) {
      ab[(static_cast<size_t>(t) * kH + h) * 2] = std::exp(-0.05f * std::fabs(nd(rng)) - 0.001f);
      ab[(static_cast<size_t>(t) * kH + h) * 2 + 1] = 1.f / (1.f + std::exp(-nd(rng)));
    }
  std::vector<float> state0(static_cast<size_t>(kH) * kD * kD);
  for (auto& x : state0) x = 0.1f * nd(rng);

  __half *d_conv, *d_khat, *d_khat_t, *d_qhat, *d_x, *d_m, *d_kk, *d_qk;
  float *d_scales, *d_ab, *d_state_ref, *d_state, *d_raw_ref, *d_g, *d_w;
  __half* d_raw;
  const size_t kq_elems = static_cast<size_t>(n_chunks) * kKH * kC * kD;
  const size_t xm_elems = static_cast<size_t>(n_chunks) * kH * kC * kC;
  HIP_OK(hipMalloc(&d_conv, conv.size() * 2));
  HIP_OK(hipMalloc(&d_khat, kq_elems * 2));
  HIP_OK(hipMalloc(&d_khat_t, kq_elems * 2));
  HIP_OK(hipMalloc(&d_qhat, kq_elems * 2));
  HIP_OK(hipMalloc(&d_kk, static_cast<size_t>(n_chunks) * kKH * kC * kC * 2));
  HIP_OK(hipMalloc(&d_qk, static_cast<size_t>(n_chunks) * kKH * kC * kC * 2));
  HIP_OK(hipMalloc(&d_x, xm_elems * 2));
  HIP_OK(hipMalloc(&d_m, xm_elems * 2));
  HIP_OK(hipMalloc(&d_scales, static_cast<size_t>(T) * kKH * 3 * 4));
  HIP_OK(hipMalloc(&d_ab, ab.size() * 4));
  HIP_OK(hipMalloc(&d_state_ref, state0.size() * 4));
  HIP_OK(hipMalloc(&d_state, state0.size() * 4));
  HIP_OK(hipMalloc(&d_raw_ref, static_cast<size_t>(T) * kH * kD * 4));
  HIP_OK(hipMalloc(&d_raw, static_cast<size_t>(T) * kH * kD * 2));
  HIP_OK(hipMalloc(&d_g, static_cast<size_t>(n_chunks) * kH * kC * 4));
  HIP_OK(hipMalloc(&d_w, static_cast<size_t>(n_chunks) * kH * kC * 4));
  HIP_OK(hipMemcpy(d_conv, conv.data(), conv.size() * 2, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_ab, ab.data(), ab.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_state_ref, state0.data(), state0.size() * 4, hipMemcpyHostToDevice));
  HIP_OK(hipMemcpy(d_state, state0.data(), state0.size() * 4, hipMemcpyHostToDevice));
  ScalesKernel<<<dim3(T, kKH), 32>>>(d_conv, d_scales, T);

  RefKernel<<<dim3(kH, kD / 64), 64>>>(d_conv, d_scales, d_ab, d_state_ref, d_raw_ref, T);
  HIP_OK(hipDeviceSynchronize());

  auto run_chunk = [&](int scan64) {
    NormKqKernel<<<dim3(kKH, n_chunks), 256>>>(d_conv, d_scales, d_khat, d_khat_t, d_qhat, d_kk, d_qk);
    ChunkPrepKernel<<<dim3(kH, n_chunks), 256>>>(d_kk, d_qk, d_ab, d_x, d_m, d_g, d_w);
    if (scan64 == 2)
      ChunkScanKernel<128><<<dim3(kH, 1), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks);
    else if (scan64)
      ChunkScanKernel<64><<<dim3(kH, kD / 64), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks);
    else
      ChunkScanKernel<32><<<dim3(kH, kD / 32), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks);
  };
  for (int variant = 0; variant < 3; ++variant) {
    HIP_OK(hipMemcpy(d_state, state0.data(), state0.size() * 4, hipMemcpyHostToDevice));
    run_chunk(variant);
    HIP_OK(hipDeviceSynchronize());
    std::vector<__half> rawh(static_cast<size_t>(T) * kH * kD); std::vector<float> raw(rawh.size()), raw_ref(rawh.size());
    HIP_OK(hipMemcpy(rawh.data(), d_raw, rawh.size() * 2, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = __half2float(rawh[i]);
    HIP_OK(hipMemcpy(raw_ref.data(), d_raw_ref, raw.size() * 4, hipMemcpyDeviceToHost));
    std::vector<float> st(state0.size()), st_ref(state0.size());
    HIP_OK(hipMemcpy(st.data(), d_state, st.size() * 4, hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(st_ref.data(), d_state_ref, st.size() * 4, hipMemcpyDeviceToHost));
    double e = 0, n = 0, es = 0, ns = 0;
    for (size_t i = 0; i < raw.size(); ++i) { const double d = raw[i] - raw_ref[i]; e += d * d; n += double(raw_ref[i]) * raw_ref[i]; }
    for (size_t i = 0; i < st.size(); ++i) { const double d = st[i] - st_ref[i]; es += d * d; ns += double(st_ref[i]) * st_ref[i]; }
    std::printf("JG=%d  output rel err %.3e   state rel err %.3e\n", variant == 2 ? 128 : variant ? 64 : 32, std::sqrt(e / n), std::sqrt(es / ns));
  }
  if (const char* only = std::getenv("GCB_ONLY")) {
    for (int i = 0; i < 3; ++i) {
      if (only[0] == '3') ChunkScanKernel<32><<<dim3(kH, kD / 32), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks);
      else ChunkScanKernel<128><<<dim3(kH, 1), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks);
    }
    HIP_OK(hipDeviceSynchronize());
    return 0;
  }
  hipEvent_t a, b;
  HIP_OK(hipEventCreate(&a)); HIP_OK(hipEventCreate(&b));
  auto time = [&](const char* name, auto fn) {
    for (int i = 0; i < 2; ++i) fn();
    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipEventRecord(a));
    for (int i = 0; i < 10; ++i) fn();
    HIP_OK(hipEventRecord(b));
    HIP_OK(hipEventSynchronize(b));
    float ms; HIP_OK(hipEventElapsedTime(&ms, a, b));
    std::printf("%-22s %8.3f ms\n", name, ms / 10);
  };
  {
    unsigned long long* d_prof; HIP_OK(hipMalloc(&d_prof, 16 * 8)); HIP_OK(hipMemset(d_prof, 0, 16 * 8));
    ChunkScanKernel<128><<<dim3(kH, 1), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks, d_prof);
    HIP_OK(hipDeviceSynchronize());
    unsigned long long pr[16]; HIP_OK(hipMemcpy(pr, d_prof, 16 * 8, hipMemcpyDeviceToHost));
    const char* names[9] = {"top+S->lds+bar", "step1", "step2", "bar", "step3", "bar", "step4", "step5", "out+bar"};
    for (int i = 0; i < 9; ++i) std::printf("  phase %-14s %8.1f cycles/chunk\n", names[i], double(pr[i]) / n_chunks);
  }
  for (int nb : {1, 8, 20, 40, 48}) {
    char nm[64]; std::snprintf(nm, sizeof nm, "P2 JG=128 blocks=%d", nb);
    time(nm, [&] { ChunkScanKernel<128><<<dim3(nb, 1), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks); });
  }
  time("reference (seq)", [&] { RefKernel<<<dim3(kH, kD / 64), 64>>>(d_conv, d_scales, d_ab, d_state_ref, d_raw_ref, T); });
  time("P0 norm", [&] { NormKqKernel<<<dim3(kKH, n_chunks), 256>>>(d_conv, d_scales, d_khat, d_khat_t, d_qhat, d_kk, d_qk); });
  time("P1 prep", [&] { ChunkPrepKernel<<<dim3(kH, n_chunks), 256>>>(d_kk, d_qk, d_ab, d_x, d_m, d_g, d_w); });
  time("P2 scan JG=32", [&] { ChunkScanKernel<32><<<dim3(kH, kD / 32), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks); });
  time("P2 scan JG=128", [&] { ChunkScanKernel<128><<<dim3(kH, 1), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks); });
  time("P2 scan JG=64", [&] { ChunkScanKernel<64><<<dim3(kH, kD / 64), 256>>>(d_conv, d_khat, d_khat_t, d_qhat, d_x, d_m, d_g, d_w, d_state, d_raw, n_chunks); });
  return 0;
}
