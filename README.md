# Strix Halo inference with the XDNA2 NPU: about 2,100 tok/s prefill

Qwen3.8 Flash-Next on AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`)
with the prefill split between the GPU and the Ryzen AI NPU.
Fork of [gufo-org/gufo](https://github.com/gufo-org/gufo) (MIT).

## Speed

| Prefill, UD-IQ4_XS | GPU only | GPU + NPU |
| --- | --- | --- |
| `gufo bench -p 8192` | 1,842 tok/s | **2,113 tok/s** (+14.7%) |
| `gufo bench -p 16384` | 1,821 tok/s | **2,117 tok/s** (+16.3%) |
| `gufo bench -p 32768` | 1,786 tok/s | **2,067 tok/s** (+15.8%) |
| Served, 54k tokens cold, MTP off | | **1,932 tok/s** |

| Decode with MTP drafting | tok/s |
| --- | --- |
| `gufo bench -n 128` | **61.1** |
| `gufo bench -n 128 -d 32768` | 93.4 |
| `gufo bench -n 128 -d 131072` | 116.8 |
| Served, code / prose / file edits / JSON / Spanish | 79 / 49 / 108 / 100 / 44 |

The `-d` rows run on the benchmark's own repetitive context, where drafts are
accepted easily; the served rows are real prompts and show what speculation
gives on each kind of text.

Three runs per prefill cell (`-r 3`), the GPU fabric clock held, EC and NPU in
performance mode, nothing else running. Two-bank mode splits each 16384-token
chunk into two 8192-token banks and runs them layer by layer in lock step: while
the NPU computes one bank's projections, the GPU works on the other. Prompt
lengths that are not a multiple of 8192 tokens fall back to a single bank for the
remainder, which is slower than the full-bank figures above. Speed also depends
on the text: a mix of code and markdown prefills 20% to 30% slower than code
alone at the same length because expert routing differs. The mode is not
bit-identical to GPU-only prefill; against GPU-only logits its last-token KL
divergence is 0.0076 on average over 10 real-text windows of 16K to 20K tokens
(the single-bank NPU path measures 0.0084), and a needle-in-a-haystack check
passes 21 of 21 at 18K, 33K and 54K tokens. `GUFO_TWO_BANK=0` turns it off.

## How the NPU is used

The NPU runs block-float 8-bit GEMMs for whole projections: the complete
linear-attention input projection (`ssm_in`) and the attention `q`/`k`/`v`
projection (`attn_qkv`), plus the shared expert's gate and up projections. The
GPU reads the NPU's output directly in its convolution, attention-preparation and
gated-delta-net kernels, so no copy or conversion pass sits between them. The
gated delta rule runs as 64-token chunks in WY form on the GPU's matrix units.

## Try it

1. Build with the NPU enabled:
   `cmake --preset release -DGUFO_ENABLE_NPU=ON -DGUFO_XRT_INCLUDE_DIR=<xrt include dir>`
   then `cmake --build --preset release --parallel 4`. The compiled NPU kernel
   (`final_4096x2560x1024_128x64x128.xclbin` and its `insts_*.txt`) and its AIE
   source are in [tools/npu](tools/npu), so nothing else has to be built.
2. Hold the GPU fabric clock at its top level (2000 MHz on this chip). This is
   required: without it the GPU and the NPU together can hang the machine.
   `sudo tools/npu/fabric-clock.sh hold` sets `power_dpm_force_performance_level`
   to `manual` and `pp_dpm_fclk` to the top level; `tools/npu/fabric-clock.sh status`
   must print `held` (`release` undoes it, a reboot also resets it). To hold it
   at every boot and after resume, run that command from a systemd oneshot unit
   wanted by `multi-user.target` and the `suspend`, `hibernate` and `hybrid-sleep`
   targets.
3. Put the NPU in performance mode:
   `sudo /opt/xilinx/xrt/bin/xrt-smi configure --pmode performance`
   (`balanced` works too and is within 1%; do not use `turbo`). The mode resets
   on every driver load, so set it after each boot.
4. Keep the NPU out of runtime suspend so its power messages cannot collide with
   other power-state writes:
   `echo on | sudo tee /sys/bus/pci/devices/<NPU PCI address>/power/control`
   (the address is the one `xrt-smi examine` prints, e.g. `0000:c7:00.1`).
5. Measure. With nothing else running:

   ```sh
   export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:/opt/rocm/lib
   GUFO_NPU_DIR=tools/npu GUFO_TWO_BANK=1 GUFO_NPU_WHOLE=1 build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3   # NPU on
   build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3                          # NPU off
   ```

   The NPU run prints `NPU: z projection of 36 layers`, `NPU: attention
   query/gate of 12 layers` and `NPU: shared expert of 48 layers` while loading.
   For the server, run `GUFO_MODEL=<first shard> GUFO_MTP=<mtp sidecar> scripts/run-gufo.sh`.
   It enables the NPU only when the fabric clock is held and the NPU is not in
   turbo mode (`GUFO_NPU=0` forces GPU-only), uses a 16384-token prefill tile
   (`GUFO_PREFILL_TILE`; 24576 is faster for 24K+ prompts but needs about 6 GB more RAM) and reads the options listed at the top of the script.

Every option is described in [docs/NPU.md](docs/NPU.md). `GUFO_NPU_SLICES` (default 6), `GUFO_NPU_GATE_SLICES` (default 12) and
`GUFO_NPU_SHEXP=0` tune or switch off parts of the offload. Without
`GUFO_NPU_DIR` the build runs GPU-only.

## Build without the NPU

```sh
cmake --preset release
cmake --build --preset release --parallel 4
./build/release/gufo serve llm --model /path/to/model.gguf
```

Needs a C++20 compiler, CMake 3.21+, Ninja, ROCm (HIP, hipBLAS, hipBLASLt,
rocBLAS, hipCUB, rocPRIM, rocWMMA) and ICU, libcurl, OpenSSL, libpng, libjpeg,
libwebp. `nix build` builds the same thing with a pinned toolchain.

## License

MIT, see [LICENSE](LICENSE). Adapted code and dependencies keep their notices in
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model
weights are not bundled.
