# Hints for RMSNorm

Each `## Hint` section is revealed one at a time by `gpu-gym hint rms_norm`.

## Hint 1

Start with one thread per row. Inside `if (row < rows)`, sum `x * x` over the row, compute `float scale = rsqrtf(sum / cols + eps);`, then write `X[...] * scale * weight[c]` in a second loop.

## Hint 2

Give each row a block: `grid = rows`, `block = 256`, `row = blockIdx.x`. Each thread sums the squares of `c = threadIdx.x, threadIdx.x + blockDim.x, ...`. Combine the partial sums in shared memory, then every thread reads the total and writes its own columns.

## Hint 3

Finish each warp's sum with `__shfl_xor_sync(0xffffffff, v, offset)` for `offset = 16, 8, 4, 2, 1`. Lane 0 of each warp writes its total to `__shared__ float partial[32]`, and after one `__syncthreads()` the first warp reduces those. Only two barriers per row remain.

## Hint 4

To read `X` only once, have each thread load its values into a small register array during the reduction, for example `float v[16]` when `cols <= 16 * blockDim.x`, and reuse them for the output. With `float4` loads each thread holds four `float4`s for `cols = 4096` and 256 threads. In Triton, one program per row with `BLOCK_SIZE = next_power_of_2(cols)` already keeps the row in registers: `x = tl.load(...)`, `ms = tl.sum(x * x, axis=0) / cols`, then store `x * tl.rsqrt(ms + eps) * w`.
