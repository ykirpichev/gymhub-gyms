# Hints for ReLU

Each `## Hint` section is revealed one at a time by `gpu-gym hint relu`.

## Hint 1

Inside the starter's `if (i < N)` block, write `C[i] = fmaxf(A[i], 0.0f);`. Use the `f` suffix on literals so the compiler does not promote to double.

PyTorch: `torch.relu` and `torch.clamp` both take `out=`. Triton: load with `mask=mask`, apply `tl.maximum(x, 0.0)`, store with the same mask.

## Hint 2

Compare the reported bandwidth with the copy peak. If you are close, the simple kernel is already near optimal. Consecutive threads read consecutive floats, so the accesses are coalesced. Changing the block size from 256 rarely helps much.

## Hint 3

Use a grid-stride loop with a fixed number of blocks, then load and store `float4` instead of `float`. Apply `fmaxf` to each of `.x`, `.y`, `.z`, `.w`. Handle the last `N % 4` elements with scalar code. Fewer memory instructions per byte helps when you are near the bandwidth limit.

## Hint 4

Mark the pointers `const float* __restrict__ A` and `float* __restrict__ C` so the compiler knows they do not overlap. Having each thread process two or four `float4` values per loop iteration, loading all of them before storing any, keeps more memory requests in flight.
