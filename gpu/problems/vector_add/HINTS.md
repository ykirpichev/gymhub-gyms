# Hints for vector add

Each `## Hint` section is revealed one at a time by `gpu-gym hint vector_add`.

## Hint 1

The starter already computes `i` and checks `if (i < N)`. Put one line inside that block that reads `A[i]` and `B[i]` and writes `C[i]`. The check matters: the `n=17` test launches 256 threads for 17 elements.

PyTorch: write into `C` with an `out=` argument or `C.copy_(...)`. `C = A + B` only rebinds a local name. Triton: `tl.load` both inputs with `mask=mask`, add, and `tl.store` with the same mask.

## Hint 2

This kernel is memory-bound. The benchmark prints the bandwidth you reached next to the copy peak. Each warp of 32 threads reads 32 consecutive floats, which the hardware merges into a few wide transactions. That is coalescing, and the one-thread-per-element layout already has it. Block size (128 to 1024) barely changes the result. Check that you are close to the peak before trying anything else.

## Hint 3

A grid-stride loop decouples the grid size from `N`. Launch a fixed number of blocks, for example a few per SM, and let each thread walk the array:

```
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
```

This saves block scheduling overhead and makes the next hint easier. You need to change the `blocks` computation in `solve`.

## Hint 4

Load four floats at once with `float4`. Cast the pointers, `reinterpret_cast<const float4*>(A)`, process `N / 4` vector elements, then handle the last `N % 4` elements with scalar code. This needs 16-byte alignment, which fresh PyTorch allocations have. Fewer, wider memory instructions often get the last few percent of bandwidth. In Triton, try a larger `BLOCK_SIZE` and `num_warps=8` in the launch.
