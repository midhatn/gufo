# NPU build: options and environment variables

The XDNA2 NPU path is optional. It is compiled in with `-DGUFO_ENABLE_NPU=ON`
(see the [README](../README.md)) and switched on at run time by pointing
`GUFO_NPU_DIR` at `tools/npu`. Without `GUFO_NPU_DIR` the same binary runs
GPU-only.

## Prerequisites at run time

| Requirement | Command |
| --- | --- |
| GPU fabric clock held at its top level | `sudo tools/npu/fabric-clock.sh hold` (`status`, `release`) |
| NPU power mode `balanced`, `performance` or `turbo` (turbo is stable with the fabric clock held and measured identical to `performance`) | `sudo xrt-smi configure --pmode performance` |
| NPU kept out of runtime suspend | `echo on \| sudo tee /sys/bus/pci/devices/<npu pci id>/power/control` |
| XRT libraries on the loader path | `LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH` |

The fabric clock changing speed while the NPU is busy can hang the machine, so
hold it before starting. The clock and power mode reset at boot.

## Easiest start

```sh
GUFO_MODEL=<first shard .gguf> GUFO_MTP=<mtp sidecar .gguf> scripts/run-gufo.sh
```

The script enables the NPU only while the fabric clock is held (any NPU power mode,
turbo included), and turns the two-bank prefill on with it (`GUFO_TWO_BANK=0` opts
out). `GUFO_PREFILL_TILE=24576` (two 12288-token banks) prefills 24K-token prompts about 2.7% faster for about 6 GB more RAM.

## Environment variables

### NPU on and off

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_DIR` | unset (NPU off) | Directory with the NPU kernel files (`tools/npu`). Setting it turns the NPU on. |
| `GUFO_NPU` | `1` | Read by `scripts/run-gufo.sh` only. `0` forces GPU-only. |

### What the NPU computes

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_SLICES` | `6` | 1024-row slices of the linear-attention `ssm_in` projection (z rows) computed on the NPU, 1 to 12. |
| `GUFO_NPU_GATE_SLICES` | `12` | 1024-row slices of the attention `attn_qkv` projection (query and gate rows, two heads per slice) on the NPU, 0 to 12. `0` turns this part off. |
| `GUFO_NPU_SHEXP` | on | `0` keeps the shared-expert gate and up projections on the GPU. |
| `GUFO_NPU_LAYERS` | all | Limit how many linear-attention layers use the NPU. |

### Prefill

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_PREFILL_TILE` | `16384` | Prefill chunk in tokens (128 to 65536). Larger fills the routed-expert tiles better and is faster: 4096 gives 1533 tok/s, 8192 1582, 16384 1606 on a 40k cold prompt (older build). Scratch memory grows with it: about 3 GiB at 8192, about 9 GiB at 16384. |

The server's `--prefill-chunk` is a separate scheduling setting (how much a
prefill runs before decoding gets a turn); `scripts/run-gufo.sh` keeps it at 4096.

### Two-bank prefill (experimental, not bit-identical)

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_TWO_BANK` | unset | `1` splits a prefill chunk of 8192 tokens or more (a multiple of 8192, or 8192 + 4096 groups) into two equal banks that run layer by layer. While one bank runs on the GPU, the NPU computes the other bank's next projection. Needs `GUFO_NPU_WHOLE=1`. `scripts/run-gufo.sh` sets both when the NPU is enabled (`GUFO_TWO_BANK=0` turns it off). |
| `GUFO_NPU_WHOLE` | unset | `1` moves the complete `ssm_in` (16 slices) and `attn_qkv` (13 slices) projections to the NPU in the two-bank path. The GPU reads the NPU output streams directly: F16 convolution rows, a chunked (64-token) gated delta rule, and fused attention preparation. Adds about 3.3 GiB of NPU weight buffers and about 240 MB of shared scratch. |

pp16384 rises from about 1,965 to about 2,120 tok/s (+8%) at the cost of block-float
rounding: final-token KL against GPU-only prefill averages 0.013 against 0.017
for the slice mode. See `EXPERIMENTS.md` for what was tried and rejected.

### Reliability and diagnostics

| Variable | Default | Effect |
| --- | --- | --- |
| `GUFO_NPU_TIMEOUT_MS` | `8000` | Time a single NPU run may take before it is treated as failed. |
| `GUFO_NPU_KEEPALIVE` | on | `0` stops the 1 s heartbeat that keeps the NPU awake. |
| `GUFO_NPU_WARMUP_MS` | `0` | Keep the NPU out of the first N ms of each prefill burst so the GPU's idle-to-full step happens alone. |
| `GUFO_NPU_ISOLATE` | unset | `1` gives the NPU private buffers and copies through the CPU. Slower; for debugging. |
| `GUFO_NPU_STATS` | unset | Print wall-clock accounting of the NPU path. |
| `GUFO_NPU_TRACE` | unset | Path of a synchronous event log, one line per NPU event. |

## Examples

NPU on, benchmark:

```sh
GUFO_NPU_DIR=tools/npu build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3
```

Same, GPU-only: leave `GUFO_NPU_DIR` unset.

Only the z rows on the NPU:

```sh
GUFO_NPU_DIR=tools/npu GUFO_NPU_GATE_SLICES=0 GUFO_NPU_SHEXP=0 build/release/gufo serve llm ...
```

Smaller prefill tile for less memory:

```sh
GUFO_PREFILL_TILE=4096 scripts/run-gufo.sh
```
