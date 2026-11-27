# Hints for sum reduction

Each `## Hint` section is revealed one at a time by `gpu-gym hint reduction`.

## Hint 1

`out` is filled with NaN before your call, so zero it first. In `solve`, call `cudaMemset(out, 0, sizeof(float))` before the launch. Then, inside the starter's `if (i < N)` block, write `atomicAdd(out, A[i]);`. That passes the tests and is slow.

PyTorch: `A.sum()` returns a 0-d tensor. Copy it into `out`. Triton: call `out.zero_()` in `solve`, sum the masked block with `tl.sum(x, axis=0)` (use `other=0.0` in the load), and `tl.atomic_add(out, total)`. That is already one atomic per block.

## Hint 2

Reduce inside each block first, then do one `atomicAdd` per block. Declare `__shared__ float buf[256];`, store `buf[tid] = (i < N) ? A[i] : 0.0f;`, and `__syncthreads()`. Then halve the active range:

```
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) buf[tid] += buf[tid + s];
    __syncthreads();
}
```

Thread 0 adds `buf[0]` to `out`. Every thread of the block must reach each `__syncthreads()`, so move the work out of the starter's `if (i < N)` block.

## Hint 3

With one element per thread, the benchmark still launches 65536 blocks. Use a grid-stride loop instead: launch a fixed number of blocks (a few per SM), and have each thread add up many elements into a register before the shared-memory step. There are fewer blocks and fewer atomics, and most of the time is spent on coalesced loads.

## Hint 4

Replace the last steps of the shared-memory tree with warp shuffles. Within a warp, `v += __shfl_down_sync(0xffffffff, v, offset)` for offsets 16, 8, 4, 2, 1 leaves the warp's sum in lane 0. Only one value per warp goes through shared memory, and the block needs one `__syncthreads()`. Load `A` as `float4` in the grid-stride loop. If you want the same result on every run, write per-block partials to a buffer and sum them in a second kernel instead of using atomics.
