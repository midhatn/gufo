# Strix Halo inference with the XDNA2 NPU

Qwen3.8 Flash-Next on AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`)
with the prefill split between the GPU and the Ryzen AI NPU.
Fork of [gufo-org/gufo](https://github.com/gufo-org/gufo) (MIT).

## Speed

| Prefill, UD-IQ4_XS | GPU only | GPU + NPU |
| --- | --- | --- |
| `gufo bench -p 16384` | 1,698 tok/s | **1,821 tok/s** (+7.2%) |
| `gufo bench -p 32768` | | **1,783 tok/s** |
| Served, 40k tokens cold, MTP and caches on | | **1,637 tok/s** |

8192-token prefill chunks, the GPU fabric clock held, NPU in performance mode,
nothing else running. Greedy output of 30 long prompts and a copy task are
identical with and without the NPU.

## How the NPU is used

The NPU runs block-float 8-bit GEMMs on parts of each layer that the GPU does
not need yet, and the GPU goes on with other work meanwhile: the `z` rows of the
linear-attention projection, the query and gate rows of 16 attention heads, and
the shared expert's gate and up projections. Each result is read back only where
the GPU first needs it.

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
   GUFO_NPU_DIR=tools/npu build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3   # NPU on
   build/release/gufo bench -m <model.gguf> -p 16384 -n 1 -r 3                          # NPU off
   ```

   The NPU run prints `NPU: z projection of 36 layers`, `NPU: attention
   query/gate of 12 layers` and `NPU: shared expert of 48 layers` while loading.
   For the server, run `GUFO_MODEL=<first shard> GUFO_MTP=<mtp sidecar> scripts/run-gufo.sh`.
   It enables the NPU only when the fabric clock is held and the NPU is not in
   turbo mode (`GUFO_NPU=0` forces GPU-only), uses an 8192-token prefill tile
   (`GUFO_PREFILL_TILE`) and reads the options listed at the top of the script.

`GUFO_NPU_SLICES` (default 6), `GUFO_NPU_GATE_SLICES` (default 8) and
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
