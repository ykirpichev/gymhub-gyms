# RMSNorm

**Level:** medium   **Teaches:** one block per row, warp shuffles, keeping a row in registers

## Task

For each row of `X`, divide by the row's root mean square, then multiply column `c` by `weight[c]`. This is the normalization used in Llama, Mistral, Qwen and most recent language models. It is layer norm without the mean and without the bias.

## Arguments

| Name     | Type           | Shape         | Role   |
| -------- | -------------- | ------------- | ------ |
| `X`      | `const float*` | `rows × cols` | input  |
| `weight` | `const float*` | `cols`        | input  |
| `Y`      | `float*`       | `rows × cols` | output |
| `rows`   | `int`          |               | size   |
| `cols`   | `int`          |               | size   |
| `eps`    | `float`        |               | scalar |

Row-major: element `(r, c)` is at `X[r * cols + c]`.

## Formula

For each row `r`:

```
ms      = (1 / cols) * sum over c of X[r, c]^2
Y[r, c] = X[r, c] / sqrt(ms + eps) * weight[c]
```

## Example

```
X      = [[1, 2, 3, 4]]     weight = [1, 1, 2, 0.5]     eps = 1e-6

ms     = (1 + 4 + 9 + 16) / 4 = 7.5      1 / sqrt(7.5) = 0.36515
Y      = [[0.36515, 0.73030, 2.19089, 0.73030]]
```

## How to think about it

Each row needs one number, the sum of its squares, before any output can be written. So the kernel is a reduction followed by an elementwise pass, like `row_softmax` and `layer_norm`, but with a single reduction instead of two.

That makes RMSNorm a good place to get the reduction really fast. The work is one multiply-add per element, so the kernel is memory-bound: the ideal is to read `X` once and write `Y` once. A simple version reads `X` twice, once to reduce and once to scale. To avoid the second read, keep the row in registers between the two steps.

If you did `layer_norm` already, start from it and delete what you do not need.

## In an interview

A memory-bound row reduction that appears in every LLM layer; interviewers look for one read of `X`.

Follow-ups to answer out loud once your kernel passes:

- What is the lower bound on time for an 8192 × 4096 FP32 input at 3 TB/s?
- How would you keep the row in registers between the reduction and the scale?
- Fuse the residual add with the norm (`x = x + residual; y = rmsnorm(x)`). What do you write back?

## This version and the other one

This is the kernel version: one row per block in CUDA or Triton, benchmarked against memory bandwidth. The LLM gym's `rms_norm` is the PyTorch version over batched CUDA tensors, where the point is the formula and its broadcasting.
