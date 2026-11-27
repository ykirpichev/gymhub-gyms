# Hints for int8_matvec

Each `## Hint` section is revealed one at a time by `gpu-gym hint int8_matvec`.

## Hint 1

Get a correct version first: one thread per row. Inside `if (n < N)`, loop over `k`, add `(float)W[n * K + k] * __half2float(x[k])` to a `float`, and write `scale[n] * acc` to `y[n]`. In PyTorch: `y.copy_((W.float() @ x.float()) * scale)`.

## Hint 2

One warp per row, as in `matvec`: `lane` walks `k = lane, lane + 32, ...`, then `__shfl_down_sync` sums the 32 partial sums and lane 0 writes `scale[n] * total`.

## Hint 3

Wide loads: when `K % 4 == 0`, read `const int* W4 = reinterpret_cast<const int*>(W + n * K);` and unpack each 32-bit word into four signed bytes with `(signed char)(w >> 8 * i)`, or use `char4`. With `K % 16 == 0`, `int4` loads move 16 weights per instruction. Keep a scalar loop for the tail so every test shape works.
