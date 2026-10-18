# Runtime-K matmul program

One XCLBIN for every reduction length that is a multiple of 64. K is not
compiled into the core: the instruction stream writes `K / 64` into each
core's `rtp` buffer and releases its lock, and the kernel flushes after that
many B tiles (`kcnt * 4` calls). Only the small instruction files differ per K.

Built (M=4096, N=1024, tile 128x64x128) with the placed program in this
directory; the XCLBIN above was generated at K=5120 and is used unchanged with
`insts_4096x2048x1024_128x64x128.txt` (Qwen3.6-35B, hidden 2048),
`insts_4096x2560x1024_128x64x128.txt` (Flash-Next) and
`insts_4096x5120x1024_128x64x128.txt` (Qwen3.8-27B). Standalone, all three pass
the numerical check at 24.3 to 29.1 TFLOPS; K=2560 is 1.2% slower than the
fixed-K program.
