# Hints for SAXPY

Each `## Hint` section is revealed one at a time by `gpu-gym hint saxpy`.

## Hint 1

Inside the starter's `if (i < N)` block, write `C[i] = alpha * A[i] + B[i];`. The compiler turns this into a single fused multiply-add (`fmaf`). Keep the math in float: write `2.0f`, not `2.0`, if you add any constants.

Triton: `alpha` is already passed into the kernel. Load `a` and `b` with the mask, compute, and store.

## Hint 2

PyTorch: `torch.mul(A, alpha, out=C)` followed by `C.add_(B)` is correct but makes two passes. The first reads `A` and writes `C` (8 bytes per element). The second reads `C` and `B` and writes `C` (12 bytes). Find a single PyTorch call that takes `alpha` and an `out=` tensor. Compare the bandwidth of both versions with `gpu-gym bench saxpy --lang pytorch`.

## Hint 3

In CUDA, the remaining gains come from moving memory efficiently. Use a grid-stride loop with a fixed number of blocks, and load `A` and `B` as `float4`. Handle the last `N % 4` elements separately. Mark the pointers `__restrict__`.

## Hint 4

Have each thread issue all of its loads before it does any math, for example two `float4` from `A` and two from `B` per loop iteration. More independent loads in flight hide memory latency better. In Triton, try a larger `BLOCK_SIZE` and `num_warps=8`.
