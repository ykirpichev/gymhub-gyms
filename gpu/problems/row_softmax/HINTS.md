# Hints for row softmax

Each `## Hint` section is revealed one at a time by `gpu-gym hint row_softmax`.

## Hint 1

Get a correct version first. In the starter, one thread owns one row. Inside the `if (row < rows)` block, write three plain loops over `c`: one for the max, one for the sum of `expf(x - max)`, and one that writes `expf(x - max) / sum`. Use `expf`, not `exp`, so the math stays in float32.

## Hint 2

One thread per row leaves most of the GPU idle: the benchmark has 128 rows, so only 128 threads do any work. Launch **one block per row** instead (`grid = rows`, `block = 256`, `row = blockIdx.x`). Each thread walks the row with a stride: `for (int c = threadIdx.x; c < cols; c += blockDim.x)`.

## Hint 3

Now the threads of a block each hold a partial max and need the block's max. That is a reduction. Write every thread's value into `__shared__ float buf[256]`, then halve the active range: `for (s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid + s]); __syncthreads(); }`. `buf[0]` is the answer. Do the same for the sum, and call `__syncthreads()` before you reuse `buf`.

## Hint 4

Faster: reduce within each warp of 32 threads using `__shfl_down_sync(0xffffffff, v, offset)` for offsets 16, 8, 4, 2, 1. Then only one value per warp goes through shared memory. An "online softmax" goes further and computes the max and the sum in a single pass. When a new max `m'` arrives, rescale the running sum by `expf(m - m')`.
