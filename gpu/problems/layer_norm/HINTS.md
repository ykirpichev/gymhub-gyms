# Hints for layer norm

Each `## Hint` section is revealed one at a time by `gpu-gym hint layer_norm`.

## Hint 1

Get a correct version first. In the starter, one thread owns one row. Inside `if (row < rows)`, write three loops over `c`: sum `x` to get the mean, sum `(x - mean) * (x - mean)` to get the variance, then write `(x - mean) * rstd * gamma[c] + beta[c]` with `float rstd = rsqrtf(var + eps);`.

## Hint 2

Launch **one block per row**: `grid = rows`, `block = 256`, `row = blockIdx.x`. Each thread walks the row with a stride, `for (int c = threadIdx.x; c < cols; c += blockDim.x)`, and keeps a partial sum. Combine the partial sums with a shared-memory tree reduction, `__syncthreads()`, and let every thread read the result from `buf[0]`. Do this twice, once for the mean and once for the variance.

## Hint 3

Reduce within each warp with `__shfl_xor_sync(0xffffffff, v, offset)` for offsets 16, 8, 4, 2, 1. After that every lane holds the warp's total. Write one value per warp to shared memory, then let the first warp reduce those. This replaces eight `__syncthreads()` rounds with two.

## Hint 4

The kernel reads `X` twice and writes `Y` once. To read it only once, keep the row in registers or shared memory: with `cols = 4096` and 256 threads, each thread holds 16 values. Or merge mean and variance into one pass with Welford: each thread keeps `(count, mean, M2)`, and two partial states combine as `delta = mb - ma; n = na + nb; mean = ma + delta * nb / n; M2 = M2a + M2b + delta * delta * na * nb / n`. Use `float4` loads when `cols % 4 == 0`.
