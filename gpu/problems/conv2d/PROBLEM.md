# 2D convolution

**Level:** medium   **Teaches:** 2D thread indexing, shared-memory tiles with halos, constant memory

## Task

Slide a small `KH × KW` filter `F` over an `H × W` image `A`. At each position where the filter fits entirely inside the image, multiply the overlapping values and add them up. Write the sums to `C`. This is a "valid" convolution with no padding and stride 1, the same as PyTorch's `conv2d` (which does not flip the filter).

## Arguments

| Name | Type           | Shape                         | Role   |
| ---- | -------------- | ----------------------------- | ------ |
| `A`  | `const float*` | `H × W`                       | input  |
| `F`  | `const float*` | `KH × KW`                     | input  |
| `C`  | `float*`       | `(H - KH + 1) × (W - KW + 1)` | output |
| `H`  | `int`          |                               | size   |
| `W`  | `int`          |                               | size   |
| `KH` | `int`          |                               | size   |
| `KW` | `int`          |                               | size   |

All three are row-major. `A[i, j]` is at `A[i * W + j]`. `C` has `OW = W - KW + 1` columns, so `C[i, j]` is at `C[i * OW + j]`. The filter is at most 15 × 15 and always fits inside the image.

## Formula

```
C[i, j] = sum over u in [0, KH), v in [0, KW) of A[i + u, j + v] * F[u, v]
```

## Example

```
A = [[1, 2, 3],        F = [[1, 2],
     [4, 5, 6],             [3, 4]]
     [7, 8, 9]]

C[0, 0] = 1*1 + 2*2 + 4*3 + 5*4 = 37
C = [[37, 47],
     [67, 77]]
```

## How to think about it

Each output is independent, so one thread per output is the natural split. Use a 2D grid: `blockIdx.y` and `threadIdx.y` pick the row, `blockIdx.x` and `threadIdx.x` pick the column. Put `x` on columns so neighbouring threads read neighbouring addresses.

Each input pixel is read by up to `KH * KW` different threads. Caches catch much of that, but you can do better by having a block load its tile of `A` into shared memory once. A block that writes a `16 × 16` patch of `C` needs a `(16 + KH - 1) × (16 + KW - 1)` patch of `A`: the tile plus a border, called the halo.

Every thread reads the same filter value at the same time. Constant memory is built for exactly that: one read, broadcast to the whole warp.

## In an interview

A tiling question in two dimensions; often turned into 'how would you do this with a matmul' (im2col).

Follow-ups to answer out loud once your kernel passes:

- Sketch the shared-memory tile with its halo for a 3×3 filter and a 16×16 block.
- What is im2col, and what does it cost in memory? When do libraries use implicit GEMM instead?
- Why do GPU convolution libraries prefer channels-last (NHWC) layouts?
