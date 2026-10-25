# Flint prefill: 700 to 4,000 tok/s

How Flint prefill went from about 700 tok/s to about 4,000 tok/s on gfx1151, what
each step bought, and which of the ideas carry over to other models. All numbers
are one machine (Strix Halo, fabric clock held at its top level), greedy decoding,
measured with the CLI (`gufo prompt -v`, "prompt=N tok in X s"). The reference
workload is a 2,197-token document-summary prompt unless noted. Timings are wall
clock for the whole prefill, interleaved A/B runs; step costs below are from the
per-stage profile (`FLINT_PROFILE=1`).

## Result

| Prompt (tokens) | Start | GPU only | GPU + NPU |
| --- | --- | --- | --- |
| 2,197 | 689-766 tok/s | 3,808 (0.577 s) | 3,990 (0.550 s) |
| 8,782 | n/a | 3,525 (2.49 s) | 3,855 (2.28 s) |
| 17,562 | n/a | 3,300 (5.32 s) | 3,607 (4.87 s) |
| 65,578 | 1,750 | 2,164 dense / 2,814 sparse | 2,287 dense |
| 131,175 | 1,115 | 1,471 dense / 2,716 sparse | n/a |

The "Start" column for 64k and 128k is the engine before the attention work
(the 2,197-token start is the first working engine). Block-sparse attention is opt-in (`FLINT_SPARSE_ATTN=1`) and approximate; the
dense rows are exact. The NPU path needs `GUFO_NPU_DIR` and a held fabric clock.

## The path, in order

Cumulative, 2,197-token prompt, GPU only until the NPU rows.

| Step | Prefill | tok/s | What changed |
| --- | --- | --- | --- |
| First working engine | 2.9-3.2 s | 689-766 | Quantized GEMMs in 8-token chunks, naive attention |
| WMMA attention, F16 qkv/shared weights | 1.44 s | 1,520 | Attention 1,420 ms to 91 ms |
| Grouped WMMA routed experts | 1.26 s | 1,743 | Experts 583 ms to 318 ms; shared expert queued before the host sync |
| One 4,096-token tile | 1.00 s | 2,206 | Prompt was split 2,048 + 149; the tail re-streamed every expert |
| F16 copies for the 7 Q5_K attn_output layers | 0.86 s | 2,550 | They ran 8 tokens at a time: 204 ms to 84 ms |
| Wide router kernel, wave top-k | 0.84 s | 2,610 | Router 61 ms to 39 ms; top-k 36 ms to about 1 ms |
| Q8_0 through the in-kernel-dequant F16 GEMM | 0.72 s | 3,057 | qkv 172 to 85 ms, shared 131 to 77 ms (K < 4096) |
| F16 router GEMM | 0.69 s | 3,182 | Router 39 ms to 16 ms |
| IQ4_XS paired gate/up kernel | 0.65 s | 3,366 | 25 layers fell back to two GEMMs: 135 to 123 ms |
| Attention LDS fix, 16 tokens per block, prefetch | 0.62 s | 3,560 | Attention 96 ms to 55 ms |
| Warm-up forward at session creation | 0.60 s | 3,643 | First-use pool sizing was inside the first prefill |
| Last layer runs the final token only | 0.59 s | 3,720 | About 9 ms |
| Transposed attention, `v_perm`, lazy rescale, `exp2` | 0.577 s | 3,808 | Attention 55 ms to 35 ms |
| NPU: shared-expert gate/up | 0.559 s | 3,930 | Hidden behind the routed experts |
| NPU: runtime-M kernel | 0.550 s | 3,990 | Smaller NPU runs, less contention |

## Ideas that should carry over

**Check the tile against the prompt before tuning kernels.** The default 2,048
tile split a 2,197-token prompt into 2,048 + 149. The 149-token tail touches
nearly every expert, so it re-streamed the whole expert set (about 0.2 s). Any
model with a chunked prefill should print per-chunk sizes first.

**Look for layers that miss the fast path.** Three of the larger wins were
"this layer silently took the slow path": Q5_K attention-output layers running in
8-token chunks, 25 IQ4_XS layers missing the paired gate/up kernel, and the
router/top-k kernels written for decode. A per-type, per-layer table of which
kernel runs is worth printing once.

**Hide the host sync on expert counts.** Grouped expert GEMMs need the per-expert
counts on the host to build tile maps. Enqueue independent work (shared expert,
the other half of the batch) between the counts copy and the sync, or the GPU
idles for 0.1-0.3 ms per layer.

**Ablate before optimizing.** Deleting a piece of a kernel (the V transpose
stores) and timing it showed 43 ms of 96 ms in one place. Reading the code had
not. The same method found that the NPU's cost to the GPU was contention, not
encoding (encode-only run: 2.660 s vs 2.659 s GPU-only; with NPU jobs running and
results ignored: 2.793 s).

**LDS bank conflicts in 2-byte transposes.** A V^T store loop where lanes run
along the head dimension hit one bank 16 times. Letting lanes run along keys and
packing neighbouring keys into 32-bit stores (one `shfl_xor` per pair) took
attention from 96 ms to 64 ms (55 ms with 16 tokens per block). Any kernel that transposes through LDS with
sub-dword stores should be checked for this.

**Transposed flash attention on WMMA.** Computing S^T = K Q^T and O^T = V^T P^T
puts one query per lane: the softmax max and sum are in registers, one cross-lane
max per tile instead of 64 shuffles, and P never goes through LDS. The
probabilities become the B operand after a lane-half swap (`v_perm` selected by
lane half). Result: 17% less attention time at every context length, bit-identical
text on a 100-token check. Then `exp2` with log2e folded into the scale (11%) and
skipping the `o *= alpha` rescale when no lane's max changed (a wave ballot, 9%).
Applicable to any GQA model with head_dim 128 on RDNA3.5.

**Count instructions, not just time.** The attention main loop was 830
instructions for 32 WMMA, 490 of them vector ALU. That explained why bigger tiles,
more waves and two-tiles-per-wave (200 of 256 VGPRs already used) did not help:
the kernel is issue-bound, so only fewer instructions help.

**Warm up on session creation.** First use of each path sizes scratch pools inside
the quantized kernels (about 10 ms at the head projection). Run a 1-token and a
64-token forward when the session is created.

**Prune the last layer.** When only the final logits are needed, the last layer's
attention, projection and FFN need one row. K and V for all rows are still
written. In two-bank mode the first bank only needs its K/V entries.

**F16 range is a correctness hazard, not just a precision one.** Sink-like tokens
produce activations above 65,504 (a routed `silu(gate) * up` of 46,880 was the
largest finite value seen; others overflowed). One inf makes a whole tile NaN
through `0 * inf` in P V, and the quantized kernels silently zero NaN inputs, so
the symptom was a wrong or `!!!!` decode, not a crash. Rule: any F16 store of an
unnormalised activation needs a scale. Flint stores routed activations divided by
64 (an `out_scale` argument on the paired gate/up kernel, default 1.0 so other
models are unchanged) and shared-expert activations either row-scaled (GPU path)
or divided by 64 (NPU path), with the factor folded into the combine weights.
Q8_1 block sums are also half precision and overflow for the same tokens.

**Validate with logits, not text.** A final-token KL against the previous build
(`KL` of 1e-8 to 4e-6 for every change kept) catches numeric regressions that a
greedy text diff hides; text diverges at near-ties for harmless reasons.

## Long context

At 64k and 128k, attention on the 14 global-attention layers is 58% of prefill
and runs at a steady 24-33 TFLOPS at every depth, so it is compute-bound, not
memory-bound (larger query blocks made it slower).

- The transposed kernel and the instruction trimming above took 128k from 117.6 s
  to 89.2 s (dense).
- **Block-sparse attention** (opt-in, approximate) on the global layers: each
  16-token group reads 2 sink blocks, the 32 blocks before it, and the 256 older
  64-key blocks with the best Quest bound (pooled key max/min, scored with WMMA
  inside the attention kernel, ordered compaction, then the normal tile loop).
  Exact while there are 256 or fewer older blocks (about the first 19k tokens).
  128k: 89.2 s to 48.3 s (2,716 tok/s).
- Retrieval check: a planted code in 34k, 63k and 111k-token haystacks at 25% and
  75% depth was found by both dense and sparse in 6 of 6 cases. No logit
  divergence measurement or multi-distractor test yet.

## NPU

- The XDNA2 kernel is a 4,096-token block-float GEMM; each projection needs two
  passes (high and residual), so for a 2k prompt it is about 6x slower than the
  GPU for the same GEMM. Only work hidden behind independent GPU work helps.
- Wins: shared-expert gate/up (hidden behind the routed experts) at every size;
  QKV for both banks of a two-bank pipeline at banks of 3,000+ tokens. Two banks
  let one bank's NPU job run during the other bank's attention and experts; the
  host sync on expert counts is hidden the same way.
- Limit: having NPU jobs in flight slows the GPU by about 5% at 8.8k tokens
  (measured by submitting jobs and ignoring the results), not through clocks.
  More NPU work stops paying.
- Out-projection (K = 4,096) on the NPU is correct but slower; it is off
  (`FLINT_NPU_OUT`).
- Balanced banks of about 2.2k tokens are far worse than full 4,096 banks (3.15 s
  vs 2.46 s at 8.8k): the NPU saturates. Chunks are cut to full 8,192.
- The runtime-M/K kernel (see `tools/npu/runtime_mk/README.md`) shrinks NPU runs
  to the prompt size: +1.5% at 2k.

## Tried and dropped

| Idea | Result |
| --- | --- |
| Second HIP stream for the shared expert | 1.7% slower (704 vs 692 ms) |
| 64-key attention tile / 16 waves per block | Slower at 2k and at 16k |
| Mask-free interior attention tiles | 17% fewer instructions but 256 VGPRs, slower |
| Pre-transposed V cache | Slower (16k: 1,169 vs 1,115 ms) |
| F16 routed down-projection output | Largest value seen 35,224; margin too thin, rejected |
| Tile sweeps for routed experts (pair 64/128/256, down 16-128) | Flat |
| Row-group variants for dense GEMMs | Flat |
| F16 weight copies for Q8_0 layers | Slower than in-kernel dequant at K < 4,096 |
| W8A8 wave64 kernel for out_proj | Slower |
| NPU for out_proj | Slower at every size |

## Decode (for completeness)

Decode went from 64 to 72 tok/s with fused kernels, stacked QKV, GQA split-K
attention and captured graphs, and to about 81 tok/s on copy-heavy text with
lossless prompt-lookup speculation. It is bandwidth-bound near 90-100 tok/s
without speculation.
