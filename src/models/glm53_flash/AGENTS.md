# GLM-5.3-Flash port notes

Numerics are pinned to upstream llama.cpp `glm5-next` (saved in
`/tmp/opencode/ref/`). Tensors are validated by `runtime/manifest.cpp`;
`tools`-style validation runs through `probe_main.cpp`.

## Artifacts

- `glm5-next` (unsloth UD-Q2_K_XL, 46 blk + MTP blk.45) = quality reference.
- `glm5next` (aj9o9 AJ-IQ2_XXS, 45 blk, MTP pruned) = runnable build:
  dense 7.12 GiB, experts 74.21 GiB.

## Forward pass (all F32, per token; hc = 4 streams)

State: `x[4*4096]` (streams), per-KDA-layer conv state `[3][3][8192]` +
delta state `[64][128][128]`, per-MLA-layer KV + indexer pool cache.

1. Embed → `x = repeat(tok_embd[tok], 4)`.
2. Each of {attn,ffn} blocks:
   - `xn = RMSNorm(x, eps=1e-5)` (flat 16384, no weight).
   - `mix = hc_*_fn^T xn` (24), `hc_scale[3]`, `hc_base[24]`.
   - `pre[c] = sigmoid(mix[c]*scale[0]+base[c])*1 + hc_eps` (c<4; eps=1e-6)
   - `post[c] = sigmoid(mix[4+c]*scale[1]+base[4+c])*2`
   - `comb[dst][src]`: rows 8..23 → `+scale[2]+base[8+·]`, reshape [4,4];
     softmax(dim=dst); then Sinkhorn: norm over src, then 19×(norm over dst,
     norm over src); every sum adds `+hc_eps` before dividing.
   - input = `Σ_s x_s·pre[s]` → RMSNorm(weight = attn_norm|ffn_norm,
     eps = layer_norm_rms_epsilon 1e-5) → block →
     `out_dst = block_out·post[dst] + Σ_s res_s·comb[dst][s]`.
3. Head: `hc_mean` = mean over streams → RMSNorm(output_norm).

## KDA layer (34 of 45)

- `Q,K,V = SiLU(causal_conv1d_k4(w?·x))` — conv weight `[4,8192]`
  (`conv_w[ch*4+k]`), zero-init state, **SiLU after conv**, no bias.
- gate: `f = ssm_f_b(ssm_f_a·x) + ssm_dt.bias` (8192);
  `g = -5·sigmoid(-f·ssm_a[h])` (ssm_a = `-exp(A_log)`, per head) — g is a
  **log-decay, per key dim**. Softplus branch only when gate_lower_bound=-inf.
- `beta[h] = sigmoid(ssm_beta·x)` (64).
- L2-normalize Q and K per head (eps 1e-6, FLA style: x/sqrt(Σx²+eps));
  q pre-scaled by 1/sqrt(128).
- Recurrence per head (S[j][i], j value-dim, i key-dim):
  `S[:,i] *= exp(g[i]); u = S k; S += ((v-u)·beta) k^T; o = S q`.
- out: `RMSNorm(o, ssm_norm)·sigmoid(ssm_g_b(ssm_g_a·x))` → `wo`.

## MLA/DSA layer (11 of 45: 3,7,…,43)

Q = q_b(RMSNorm_qa(q_a·x)), KV = kv_a→(c_kv[512], k_pe[64]);
heads 64, key_len 192 (nope 128 from kv 512 split? see graph — no rope dims:
`rope.dimension_count = 0`), no rotary anywhere.
Indexer: q via `indexer.attn_q_b`, k = `indexer.attn_k`, L2 norm k
(`indexer.k_norm` w/b), scores `relu(q·k)·proj` summed over 32 heads;
k-pool=4: score pools of 4 tokens, select top `2048/4=512` pools + tail,
`indexer_compressor_gate/ape` compress pool keys. Additive mask → softmax in
the MLA attention over selected positions. (Detail from
`build_kpool_select`/`build_dsa_layer` — implement after KDA golden.)

## FFN

- Dense 0–2: PAR silu with **clamp 10**:
  `min(gate,10)·sigmoid(min(gate,10))·clamp(up,±10)` → ffn_down.
- MoE 3–44: router F32 `4096×288`, `+exp_probs_b`, sigmoid gating;
  top-8; `weights_norm` renorm; `×2.5`; expert swiglu clamp 10; add shared
  expert (same clamped swiglu).

## Quants

MMQ tier (`src/models/deepseek_v4_flash/kernels/rocm/mmq/`, vendored
llama.cpp) covers every GLM type incl. IQ2_XXS/IQ2_XS/IQ3_XXS/Q2_K/Q3_K.
CPU dequant in `src/core/quant/ggml_dequant.cpp` (no IQ2_XXS yet — needed
for the CPU reference only).
