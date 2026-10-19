# Qwen3.6 35B-A3B quality

Reference: llama.cpp on the CPU with the same GGUF. Each case prefills a real-text
window (repository documentation or C++ source), feeds the next 1,024 tokens one
at a time and compares the next-token distribution at 512 positions per window,
three windows (1,536 positions).

| Build | KL vs llama.cpp | Top-1 | Perplexity change |
| --- | --- | --- | --- |
| GPU only | 0.0034 | 95.8% | +0.002% |
| GPU + NPU | 0.0040 | 95.9% | -0.026% |
| fp32 router GEMM (`GUFO_QWEN_ROUTER_SPLIT=0`) | 0.0033 | 95.3% | +0.28% |
| per-token prefill (`GUFO_QWEN_MOE_PREFILL=0`) | 0.0035 | 95.4% | +0.31% |

Expert routing has near-ties, so any numeric change (a different summation
order, the NPU, the router GEMM) moves 0.002 to 0.003 KL between builds without
changing quality: the four rows differ from each other by about as much as each
differs from the reference. Perplexity is the quality measure; top-1 agreement
between two builds is not.

At 8,192 tokens the NPU build agrees with GPU-only at KL 0.004 and 0.001 on a
documentation and a source window; lock-step chunk pairs are bit-identical to
the one-chunk-at-a-time order.

## Speculative decoding

Verification uses the same kernels as decode, so greedy MTP, DFlash2 and auto
output equals plain decode except at near-tie tokens: batched and single-token
paths differ by a logit difference of 0.2 to 1.3, all argmax equal on a
16-row code/prose check, and three of six served workloads match plain
byte for byte (the rest diverge at the first near-tie). A batch of up to 8 rows
of verification logits agrees with sequential decode within that tolerance.

MTP priming matches the per-token path: drafted, accepted and step counts are
identical on a 2.3K-token prompt.

## Restart

MTP and auto sessions restore from `--cache-disk` after a restart with identical
output on a 2,945-token prompt and its continuation.
