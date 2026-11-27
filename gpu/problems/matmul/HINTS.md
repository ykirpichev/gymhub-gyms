# Hints for matrix multiply

Each `## Hint` section is revealed one at a time by `gpu-gym hint matmul`.

## Hint 1

In the starter, `m` is the row and `n` the column of `C`. Inside `if (m < M && n < N)`, loop `k` from 0 to `K`, accumulate `A[m * K + k] * B[k * N + n]` into a local `float sum = 0.0f`, then write `C[m * N + n] = sum`. `threadIdx.x` moves along `n`, so a warp reads consecutive elements of `B`, and all its threads read the same `A` element, which is a broadcast.

PyTorch: `torch.matmul` takes `out=`.

## Hint 2

Tile through shared memory. Each 16 × 16 block computes a 16 × 16 tile of `C` and walks along `K` in steps of 16. In each step, every thread loads one element of `A` and one of `B` into `__shared__ float As[16][16], Bs[16][16];`, calls `__syncthreads()`, does 16 multiply-adds from shared memory, and calls `__syncthreads()` again before the next load. This cuts global loads by a factor of 16. Load 0 for elements outside the matrix. The `17x13 @ 13x19` test has partial tiles in all three dimensions.

## Hint 3

Triton: accumulate in `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`, loop over `K` in steps of `BLOCK_K`, load masked tiles of `a` and `b`, and add `tl.dot(a_tile, b_tile)`. By default `tl.dot` uses TF32 for float32 inputs, and that fails this problem's `1e-3` tolerance. Pass `input_precision="ieee"`.

CUDA: shared-memory reads now limit the tiled kernel. Give each thread a small patch of `C`, for example 4 × 4 or 8 × 8 outputs kept in registers. For each `k` in the tile, load a few values of `As` and `Bs` into registers and do an outer product. A block of 256 threads then covers a 64 × 64 or 128 × 128 tile of `C`. Change the launch in `solve` to match.

## Hint 4

Further steps, roughly in order of payoff:

- Load global tiles with `float4`.
- Store the `A` tile transposed in shared memory, so the inner loop reads both tiles along rows with no bank conflicts.
- Double-buffer: load the next tile, with `cp.async` or plain registers, while computing on the current one.

Tensor cores (WMMA with TF32) are much faster, but like Triton's default they may exceed the tolerance. Check the error before relying on them.
