# Matrix-vector multiply

**Level:** medium   **Teaches:** one warp or block per row, reductions, memory-bound GEMV

## Task

Multiply the matrix `A` by the vector `x` and write the result to `y`. Each output is the dot product of one row of `A` with `x`.

## Arguments

| Name | Type           | Shape   | Role   |
| ---- | -------------- | ------- | ------ |
| `A`  | `const float*` | `M × K` | input  |
| `x`  | `const float*` | `K`     | input  |
| `y`  | `float*`       | `M`     | output |
| `M`  | `int`          |         | size   |
| `K`  | `int`          |         | size   |

`A` is row-major. Element `(r, k)` is at `A[r * K + k]`.

## Formula

```
y[r] = sum over k of A[r, k] * x[k]    for r = 0 .. M-1
```

## Example

```
A = [[1, 2, 3],
     [4, 5, 6]]     M = 2, K = 3
x = [1, 0, -1]

y[0] = 1*1 + 2*0 + 3*(-1) = -2
y[1] = 4*1 + 5*0 + 6*(-1) = -2
y = [-2, -2]
```

## How to think about it

Every element of `A` is used exactly once, so the kernel cannot reuse it. The time is set by how fast you can stream `M * K` floats from memory. The arithmetic is only 2 flops per 4 bytes. That makes matvec memory-bound, unlike matmul.

One thread per row is correct, but it reads memory badly: neighbouring threads read addresses `K` floats apart, so a warp's loads are not coalesced. Turn it around. Let a group of threads share one row, walk along it together so their loads are consecutive, and then add up their partial sums. A warp of 32 threads per row is a good group size. `x` is small and read by every row, so it stays in cache.

## In an interview

Matrix-vector product is memory-bound: the matrix is read once. Asked to see whether you map warps to rows for coalescing.

Follow-ups to answer out loud once your kernel passes:

- One thread per row or one warp per row? Which loads coalesce in a row-major matrix?
- Why is LLM decoding at batch size 1 essentially a sequence of matvecs, and what bounds its speed?
- How would the kernel change for a column-major matrix?
