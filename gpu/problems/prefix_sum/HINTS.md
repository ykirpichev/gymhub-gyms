# Hints for prefix sum

Each `## Hint` section is revealed one at a time by `gpu-gym hint prefix_sum`.

## Hint 1

Get a correct version first. Launch one thread (`scan_kernel<<<1, 1>>>`) and write the plain loop: `float s = 0; for (int i = 0; i < N; i++) { s += A[i]; B[i] = s; }`. It is slow, but it passes the tests and gives you a baseline to beat.

## Hint 2

Scan inside one block with the Hillis-Steele pattern. Load your block's chunk into `__shared__ float buf[BLOCK]` (use 0 past the end of `A`). Then for `offset = 1, 2, 4, ...` up to `blockDim.x`: read `t = (tid >= offset) ? buf[tid - offset] : 0`, call `__syncthreads()`, add `t` to `buf[tid]`, and call `__syncthreads()` again. The two barriers keep a thread from reading a value that another thread is updating in the same round.

## Hint 3

For more than one block, keep a scratch array of one float per block. A device pointer can stay alive between calls: `static float* sums = nullptr; static int cap = 0;` and `cudaMalloc` it only when `cap` is too small. Kernel 1 scans each chunk into `B` and writes the chunk total to `sums[blockIdx.x]`. Kernel 2 scans `sums` (one block is enough when there are at most a few thousand chunks, or recurse). Kernel 3 adds `sums[blockIdx.x - 1]` to every element of chunk `blockIdx.x > 0`.

## Hint 4

To approach copy speed, cut the work and the memory passes. Let each thread load 4 to 16 consecutive values and scan them in registers, so only one value per thread goes through the block scan. Replace the shared-memory rounds with warp shuffles: `v += (lane >= d) ? __shfl_up_sync(0xffffffff, v, d) : 0` for `d = 1, 2, 4, 8, 16`, then scan the 8 warp totals. A single-pass scan with "decoupled look-back" (each block waits for the running total published by the block before it) reads `A` and writes `B` only once.
