# Hints for GELU

Each `## Hint` section is revealed one at a time by `gpu-gym hint gelu`.

## Hint 1

Inside `if (i < N)`, read `float x = X[i];`, compute `u = 0.7978845608f * (x + 0.044715f * x * x * x)`, and write `Y[i] = 0.5f * x * (1.0f + tanhf(u));`. Keep the `f` suffix on every constant. A bare `0.5` is a double and pulls the math into float64, which is slow on most GPUs.

## Hint 2

If you write tanh yourself, use `tanh(u) = 1 - 2 / (expf(2u) + 1)`. For large `u`, `expf(2u)` becomes `inf` and the formula gives `1 - 0 = 1`, which is still correct. The form `(e^u - e^-u) / (e^u + e^-u)` gives `inf / inf = NaN` there, so avoid it.

## Hint 3

The kernel is memory-bound, so wider memory operations help most. Let each thread load a `float4` with `reinterpret_cast<const float4*>(X)[j]`, apply GELU to `.x .y .z .w`, and store a `float4`. Launch `N / 4` threads for the vector part and handle the last `N % 4` elements with scalar code.

## Hint 4

A grid-stride loop over a fixed number of blocks (a few per SM) lets each thread process several `float4`s and keeps many loads in flight. `__expf` is a faster, slightly less accurate `expf`. It is accurate enough here, but check the result with `gpu-gym test` after you switch.
