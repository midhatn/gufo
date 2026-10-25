# Flint

Mixture-of-experts text model (49 layers, hidden 2048, 32 query / 4 KV heads of
128, 256 experts with 6 routed plus a shared expert, 2 leading dense layers,
sliding-window attention of 513 on 35 of the layers) on gfx1151. The tested weights are
`flint-UD-Q4_K_XL.gguf` (25 GB; Q8_0 attention (Q5_K in 7 layers), Q4_K / Q5_K / IQ3_S / IQ4_XS
experts, BF16 router).

[Prefill optimization history](PREFILL.md) · NPU kernel: `tools/npu/runtime_mk/`

## Run

```sh
gufo prompt -m flint-UD-Q4_K_XL.gguf -f prompt.txt -n 128 -t 0 -v
```

`--cpu` runs the slow CPU reference, `--speculative off` turns off the lossless
prompt-lookup speculation (on by default in `gufo prompt`).

Serving: `GUFO_KIND=flint ~/run-gufo.sh` (or `gufo serve llm -m flint-UD-Q4_K_XL.gguf`).
Continuation snapshots (RAM and disk) restore the KV rows and last logits; a
restart restores an 8,790-token prompt from disk in 1.7 s instead of a 3.0 s
prefill. The server samples with the model's recommended settings unless the
request overrides them. Tool calls are not supported yet (rejected with an error);
`gufo serve` decodes one token per step, without lookup speculation.

## Speed (GPU, interleaved runs, one machine)

| | Prefill | Decode |
| --- | --- | --- |
| 2,197-token prompt | 3,808 tok/s (3,990 with NPU) | 72 tok/s (68 at 2.2k depth) |
| 8,782 tokens | 3,525 (3,855 with NPU) | |
| 131,175 tokens | 1,471 dense, 2,716 sparse | |

`gufo bench -p 2048,4096,8192 -n 128`: prefill 3,882 / 3,933 / 3,724 GPU,
4,150 / 4,007 / 4,143 with the NPU; decode 73.6. Decode on generated text, greedy
(plain / lookup): code 72.6 / 74.9, prose 72.6 / 72.3, JSON 72.6 / 73.3; HTTP
serve decode 72.8. Lookup only helps when the output copies earlier text; there is
no draft model for Flint. Decode reaches about 81 tok/s on copy-heavy text. Details and the
steps that got here are in [PREFILL.md](PREFILL.md).

## Options (environment)

| Variable | Effect |
| --- | --- |
| `FLINT_PROFILE=1` | Per-stage GPU timing for prefill and decode |
| `FLINT_SPARSE_ATTN=1` | Block-sparse attention on the global layers; approximate, exact below about 19k tokens |
| `FLINT_SPARSE_TOPK=N` | Older 64-key blocks kept per 16-token group (default 256, at most 400) |
| `GUFO_NPU_DIR=<repo>/tools/npu` | Use the NPU (shared-expert gate/up, and QKV for banks of 3,000+ tokens); needs the fabric clock held and a binary named `gufo*` |
| `FLINT_NPU_BANKS=0` | NPU without the two-bank pipeline |
| `FLINT_NPU_FIXED_M=1` | Old fixed 4,096-token NPU kernel instead of the runtime-M one |
| `FLINT_NPU_OUT=<mask>` | Attention output projection on the NPU for banks (bit 0, bit 1); correct but slower, off by default |

## Numerical notes

- Routed and shared-expert activations are stored in F16 divided by 64 (the
  combine multiplies it back); unscaled values from sink-like tokens exceed
  65,504 and turn a whole attention tile into NaN. See PREFILL.md.
- The NPU path is not bit-identical to the GPU path: final-token KL against
  GPU-only was 1e-8 to 3e-6 in every test, with the same top-1 token.

## Quality

HumanEval pass@1, greedy, `--think off`, one fenced Python block, GPU engine:
**121 / 164 = 73.8%** with a 1,536-token cap. Most failures are not wrong code:
the model reasons at length even with `--think off` and many replies hit the cap
before closing the reasoning block (`SyntaxError: unterminated string literal` on
raw reasoning text). A rerun of the truncated problems with a 4,096-token cap is
pending. Two model instances fit in GPU memory at once; a third fails at load.

## Status

Prefill and decode are verified against a CPU reference and by greedy text on a
2k prompt. Not done: tool use in the chat template, model tests under
`tests/`, and the qwen/glm regression run for the two
shared Flash-Next kernel edits (an IQ4_XS case in the paired routed launcher, an
`out_scale` argument defaulting to 1.0, and a 1024/2048-row dense F16 config).
