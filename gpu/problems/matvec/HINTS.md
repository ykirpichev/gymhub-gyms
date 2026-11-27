# Hints for matvec

Each `## Hint` section is revealed one at a time by `gpu-gym hint matvec`.

## Hint 1

Get a correct version first. In the starter, one thread owns one row. Inside `if (row < M)`, loop `k` from 0 to `K`, add `A[row * K + k] * x[k]` to a float, and write it to `y[row]`. It passes the tests but is slow, because a warp's 32 loads land in 32 different rows.

## Hint 2

Give each row one warp instead. With 256 threads per block, a block handles 8 rows: `warp = threadIdx.x / 32`, `lane = threadIdx.x % 32`, `row = blockIdx.x * 8 + warp`. Each lane walks the row with a stride: `for (int k = lane; k < K; k += 32)`. Now the warp reads 32 consecutive floats per step. Change the launch to `(M + 7) / 8` blocks.

## Hint 3

Each lane now holds a partial sum. Add them across the warp with shuffles, no shared memory needed: `for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);`. Lane 0 then holds the total and writes `y[row]`. Every lane must reach the shuffle, so do not return early for lanes that had no work.

## Hint 4

To get closer to peak bandwidth, load `float4`s from the row: each lane reads 4 consecutive floats per step and the warp covers 128 floats. This needs `K % 4 == 0` so every row starts 16-byte aligned. Keep a scalar path for other `K`. Unrolling the loop so each lane has 2 to 4 loads in flight before it adds also helps hide memory latency.
