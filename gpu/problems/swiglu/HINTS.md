# Hints for fused SwiGLU

Each `## Hint` section is revealed one at a time by `gpu-gym hint swiglu`.

## Hint 1

With one thread per output, `r = idx / cols` and `c = idx % cols`. Read `float g = X[r * 2 * cols + c];` and `float u = X[r * 2 * cols + cols + c];`, then write `Y[idx] = g / (1.0f + expf(-g)) * u;`.

## Hint 2

Use a 2D grid so you never divide: `c = blockIdx.x * blockDim.x + threadIdx.x`, `r = blockIdx.y`, with `grid = dim3((cols + 255) / 256, rows)`. Remember to change the launch in `solve` too.

## Hint 3

When `cols % 4 == 0`, both halves of a row start at a 16-byte boundary, so each thread can load one `float4` of gates and one of up values and store a `float4`.

## Hint 4

In Triton use a 2D program grid `(rows, cdiv(cols, BLOCK))`. Load the gate from `x + r * 2 * cols + offs` and the up value from the same pointer plus `cols`. `tl.sigmoid(g)` gives `1 / (1 + exp(-g))`.
