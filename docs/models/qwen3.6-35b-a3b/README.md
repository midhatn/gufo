# Qwen3.6 35B-A3B

Mixture-of-experts text model (256 experts, 8 routed plus a gated shared expert,
30 linear-attention and 10 full-attention layers) on gfx1151. The tested weights
are **UD-Q4_K_XL** (21.5 GiB). Images are not supported by the loader.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --local-dir models/qwen3.6-35b
nix build
MODEL=models/qwen3.6-35b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp --mtp-model embedded
```

| Mode | Selection | Behavior |
| --- | --- | --- |
| AR | No speculative option | Target-only generation. |
| MTP | `--speculative mtp --mtp-model embedded` | The MTP head stored in the checkpoint (`blk.40`) drafts up to three tokens. |
| DFlash2 | `--speculative dflash2 --dflash-model PATH` | Companion draft model fed by several target layers. |
| Auto | `--speculative auto --dflash-model PATH --mtp-model embedded` | Both drafts; each request uses whichever commits more tokens per second. |

DFlash2 is faster on code and JSON, MTP on prose. Sampled requests always use
DFlash2, the draft that reports its proposal distribution.

## Execution

- **Prefill** groups the tokens by expert and runs the routed experts as F16
  matrix-core GEMMs (Q6_K expert tensors are re-encoded to Q8_0 at load). The
  router runs in fp32 accuracy through one split-bf16 GEMM, and a 16-over-2 head
  WMMA kernel handles attention.
- **NPU prefill** (`GUFO_NPU_DIR`, see [NPU options](../../NPU.md)) computes the
  leading columns of the linear-attention and full-attention projections.
- **Decode** runs the whole feed-forward on the device in a captured graph:
  top-k routing, quantized expert vector kernels and the gated shared expert.
- **Verification** of drafted tokens batches up to 64 rows through the same
  expert kernels.
