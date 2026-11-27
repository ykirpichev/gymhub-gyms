# Hints for reverse

Each `## Hint` section is revealed one at a time by `gpu-gym hint reverse`.

## Hint 1

Inside the `if (i < N)` block, write one line: `B[i] = A[N - 1 - i];`. The guard matters because `N` is often not a multiple of the block size.

## Hint 2

That version is already coalesced. A warp touches 32 consecutive floats of `A` (in reverse) and 32 consecutive floats of `B`. Run `gpu-gym bench reverse --lang cuda` and compare the GB/s to the copy peak before you change anything.

## Hint 3

To move more data per instruction, let each thread handle 4 floats with `float4` loads and stores. Thread `t` writes `B[4t .. 4t+3]` from `A[N-4-4t .. N-1-4t]`, with the 4 values swapped (`w, z, y, x`). The catch: `float4` needs a 16-byte aligned address, and `N - 4 - 4t` is only a multiple of 4 when `N` is. Use the vector path when `N % 4 == 0` and fall back to scalar code otherwise, or handle the few unaligned elements at the ends separately.

## Hint 4

A grid-stride loop (`for (i = start; i < N; i += gridDim.x * blockDim.x)`) with a fixed grid of a few blocks per SM cuts launch overhead and keeps enough loads in flight. Reading through `__ldg` or marking `A` as `const float* __restrict__` lets the compiler use the read-only cache path.
