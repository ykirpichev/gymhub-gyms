# Hints for histogram

Each `## Hint` section is revealed one at a time by `gpu-gym hint histogram`.

## Hint 1

Get a correct version first. Zero `counts` in `solve` before the launch: `cudaMemset(counts, 0, bins * sizeof(float));`. In the kernel, each thread reads one value and does `atomicAdd(&counts[(int)A[i]], 1.0f);`. Do not use `+=`: two threads can read the same old count, and one update is lost.

## Hint 2

With 256 bins and 16 million elements, about 65 thousand threads fight over each bin. Give each block its own histogram: `extern __shared__ float local[];` sized `bins` floats (pass `bins * sizeof(float)` as the third launch parameter). Zero it, `__syncthreads()`, do the `atomicAdd` into `local`, `__syncthreads()`, then let thread `t` add `local[t]` (and `local[t + blockDim.x]`, ...) into `counts` with one global `atomicAdd` per bin.

## Hint 3

Launch a fixed number of blocks, a few per SM (for example 4 * the SM count, or just 256), and let each thread walk the array with a grid stride: `for (i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)`. Each block then merges its histogram once, instead of once per 256 elements.

## Hint 4

Read four values at a time with `float4` loads to cut the number of memory instructions; handle the last `N % 4` elements separately. Counting in `unsigned int` in shared memory (`atomicAdd` on integers) is a little faster than float. Convert to float when you merge. When every value falls in the same bin, even shared-memory atomics serialize; giving each warp its own copy of the histogram spreads the load.
