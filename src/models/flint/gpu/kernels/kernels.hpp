#ifndef GUFO_MODELS_FLINT_GPU_KERNELS_KERNELS_HPP_
#define GUFO_MODELS_FLINT_GPU_KERNELS_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Flint-private HIP launchers. Activations are row-major float32
/// [tokens][dim]; every launch is asynchronous on `stream`. Positions are read
/// from device memory so a captured decode graph replays at any depth.
namespace gufo::models::flint::gpu::rocm {

/// Staged C streams of an NPU GEMM: the high pass and the low (residual) pass
/// of up to eight 1024-column slices.
struct NpuStreams {
  const std::uint8_t* hi[8];
  const std::uint8_t* lo[8];
};
/// silu(gate) * up * scale from the staged C streams (high and residual
/// passes of the gate and up GEMMs) as F16 rows [tokens][valid_cols].
void NpuSwigluScaled(const void* gate_hi, const void* up_hi,
                     const void* gate_lo, const void* up_lo, __half* out,
                     std::uint32_t tokens, std::uint32_t cols,
                     std::uint32_t valid_cols, float scale, hipStream_t stream);
/// Block-float A streams (high and residual) of float rows [tokens][ldx];
/// tokens a multiple of 128, k_slice a multiple of 64.
void NpuEncodeBoth(const float* x, std::size_t ldx, std::uint32_t tokens,
                   std::uint32_t k_slice, void* hi, void* lo,
                   hipStream_t stream);
void NpuDecodeFused(const NpuStreams& streams, std::uint32_t slices, float* out,
                    std::uint32_t tokens, std::uint32_t cols, std::size_t pitch,
                    hipStream_t stream);

/// out[t] = dequantized Q8_0 row tokens[t] of the embedding table.
void EmbedQ8(const void* table, const std::int32_t* tokens, float* out,
             std::uint32_t n_tokens, std::uint32_t hidden, hipStream_t stream);

/// RMSNorm of `groups` equal slices per row; gamma (dim / groups floats) is
/// shared by every slice and may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// NEOX rotation: pairs (i, i + head_dim / 2) of every head at position
/// *start_pos + t, in place.
void RopeNeoX(float* x, std::uint32_t n_tokens, std::uint32_t heads,
              std::uint32_t head_dim, const std::uint32_t* start_pos,
              float base, hipStream_t stream);

/// Stores K and V rows as F16 at cache row *start_pos + t.
void StoreKv(const float* k, const float* v, __half* k_cache, __half* v_cache,
             std::uint32_t n_tokens, std::uint32_t row_dim,
             const std::uint32_t* start_pos, hipStream_t stream);

/// Causal attention with per-head sink logits. `window` > 0 limits each query
/// to its last `window` keys. out is [t][heads][head_dim].
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float* out, std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t head_dim,
               std::uint32_t window, hipStream_t stream);

/// Prefill causal attention on the matrix cores (F16 query rows): 16-row WMMA
/// tiles pack two tokens of one KV head's eight query heads. Needs head_dim 128
/// and a GQA group of 8; returns false (launching nothing) otherwise.
bool AttentionWmma(const __half* q, const __half* k_cache,
                   const __half* v_cache, const float* sinks, float* out,
                   std::uint32_t n_tokens, const std::uint32_t* start_pos,
                   std::uint32_t heads, std::uint32_t kv_heads,
                   std::uint32_t head_dim, std::uint32_t window,
                   hipStream_t stream);

/// The same over a block-sparse key set (global layers): each 16-token group
/// reads two sink blocks, the 32 blocks before its first token and its own
/// span, plus the `top_blocks` older 64-key blocks with the best Quest bound
/// (pooled max/min rows from PoolKeyBlocks, nblk <= 4096 blocks per head).
/// Exact when no more than `top_blocks` older blocks exist.
bool AttentionWmmaSparse(const __half* q, const __half* k_cache,
                         const __half* v_cache, const __half* kmm,
                         std::uint32_t nblk, std::uint32_t top_blocks,
                         const float* sinks, float* out, std::uint32_t n_tokens,
                         const std::uint32_t* start_pos, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t head_dim,
                         hipStream_t stream);

/// Recomputes the pooled key bounds of every 64-key block that positions
/// [start, start + n_tokens) touch. `start_host` mirrors *start_pos.
void PoolKeyBlocks(const __half* k_cache, __half* kmm,
                   const std::uint32_t* start_pos, std::uint32_t start_host,
                   std::uint32_t n_tokens, std::uint32_t kv_heads,
                   std::uint32_t nblk, hipStream_t stream);

/// Decode-width causal attention (1..8 tokens): the key range of every query is
/// cut into `splits` slices that run in parallel and are merged with the sink
/// logit. The merged rows go to `out_q8` as Q8_1 rows when it is not null,
/// else to `out`. `partial` holds n_tokens * heads * splits * (head_dim + 2)
/// floats.
void AttentionSplit(const float* q, const __half* k_cache,
                    const __half* v_cache, const float* sinks, float* out,
                    void* out_q8, float* partial, std::uint32_t n_tokens,
                    const std::uint32_t* start_pos, std::uint32_t heads,
                    std::uint32_t kv_heads, std::uint32_t head_dim,
                    std::uint32_t window, std::uint32_t splits,
                    hipStream_t stream);

/// x[t] += delta[t] (when delta is not null), then xn[t] = rmsnorm(x[t]) *
/// gamma and, when q8 is not null, xn's Q8_1 rows (block_q8_1 layout, 36 bytes
/// per 32 values) in q8. dim must equal 8 * the block size, at most 2048.
void ResidualNormQuant(float* x, const float* delta, const float* gamma,
                       float* xn, void* q8, std::uint32_t n_tokens,
                       std::uint32_t dim, float eps, hipStream_t stream);

/// ResidualNormQuant whose residual update is the MoE combine
/// sum_s weights[t][s] * expert_scale[t * k + s] * expert_out[t * k + s]
/// + shared_scale[t] * shared[t]; either scale may be null (meaning 1).
void MoeCombineNormQuant(float* x, const float* expert_out,
                         const float* weights, const float* expert_scale,
                         const float* shared, const float* shared_scale,
                         std::uint32_t k, const float* gamma, float* xn,
                         void* q8, std::uint32_t n_tokens, std::uint32_t dim,
                         float eps, hipStream_t stream);

/// gate[r][i] = silu(gate[r][i]) * up[r][i] for `rows` rows of `cols`, divided
/// by a power-of-two scale[r] when the row maximum exceeds a limit that keeps
/// the half-precision block sums of a Q8_1 quantization finite (scale[r] is 1
/// otherwise). The next projection's output is then rescaled exactly.
void SwigluScaled(float* gate, const float* up, float* scale,
                  std::uint32_t rows, std::uint32_t cols, hipStream_t stream);

/// Per-head QK RMSNorm, NEOX rope (when `rope`) and the KV store in one launch.
/// Each source is (pointer, row stride in floats): q holds `heads` heads, k and
/// v `kv_heads` heads of `head_dim`. Q lands in q_out [t][heads * head_dim];
/// K and V land in the F16 caches at row *start_pos + t.
struct QkvSources {
  const float* q;
  std::uint32_t q_stride;
  const float* k;
  std::uint32_t k_stride;
  const float* v;
  std::uint32_t v_stride;
};
void QkvPrepare(const QkvSources& src, const float* q_gamma,
                const float* k_gamma, float* q_out, __half* k_cache,
                __half* v_cache, std::uint32_t n_tokens, std::uint32_t heads,
                std::uint32_t kv_heads, std::uint32_t head_dim,
                const std::uint32_t* start_pos, float base, float eps,
                bool rope, hipStream_t stream, __half* q_half = nullptr);

/// Sigmoid gating, top-k, optional renormalization and scale per token.
void SigmoidTopK(const float* logits, std::int32_t* ids, float* weights,
                 std::uint32_t n_tokens, std::uint32_t n_experts,
                 std::uint32_t k, bool normalize, float scale,
                 hipStream_t stream);

/// out[t][e] = dot(bf16 w[e], x[t]).
void RouterBf16(const void* w, const float* x, float* out,
                std::uint32_t n_tokens, std::uint32_t n_experts,
                std::uint32_t k, hipStream_t stream);

/// gate[i] = silu(gate[i]) * up[i].
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);

/// x[t] += sum_s weights[t][s] * expert_out[t * k + s] + shared[t].
void MoeCombine(float* x, const float* expert_out, const float* weights,
                const float* shared, std::uint32_t n_tokens, std::uint32_t k,
                std::uint32_t dim, hipStream_t stream);

/// out[t][i] = half(in[t][i]) with row pitch `ldx` halves.
void NarrowRows(const float* in, __half* out, std::uint32_t n_rows,
                std::uint32_t k, std::uint32_t ldx, hipStream_t stream);

/// Expands a Q8_0 matrix of `count` elements (a multiple of 32) to F16.
void ExpandQ8ToHalf(const void* q8, __half* out, std::size_t count,
                    hipStream_t stream);

/// x[i] += y[i].
void AddInPlace(float* x, const float* y, std::size_t count,
                hipStream_t stream);

}  // namespace gufo::models::flint::gpu::rocm

#endif  // GUFO_MODELS_FLINT_GPU_KERNELS_KERNELS_HPP_
