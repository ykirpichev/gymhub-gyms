# SwiGLU feed-forward block

**Level:** easy   **Teaches:** gated linear units, fusing two matmuls into one, where an MLP spends its FLOPs

## Task

Compute the feed-forward block used by Llama, Mistral and Qwen: project `x` up twice, gate one projection with SiLU, multiply, and project back down. Return a new tensor.

## Arguments

| Name     | Type           | Shape       | Role                        |
| -------- | -------------- | ----------- | --------------------------- |
| `x`      | `torch.Tensor` | `[..., D]`  | input, float32, CUDA        |
| `w_gate` | `torch.Tensor` | `[F, D]`    | gate projection             |
| `w_up`   | `torch.Tensor` | `[F, D]`    | up projection               |
| `w_down` | `torch.Tensor` | `[D, F]`    | down projection             |
| returns  | `torch.Tensor` | `[..., D]`  | output                      |

## Formula

```
silu(z) = z * sigmoid(z)
h       = silu(x @ w_gate^T) * (x @ w_up^T)       # [..., F]
y       = h @ w_down^T                             # [..., D]
```

The weights are stored like `nn.Linear` weights: `[out_features, in_features]`.

## Example

```
x = [[1, -1]]   w_gate = [[1, 0], [0, 1]]   w_up = [[2, 0], [0, 2]]   w_down = [[1, 1], [0, 1]]
x @ w_gate^T = [1, -1]          silu = [0.7311, -0.2689]
x @ w_up^T   = [2, -2]          h    = [1.4621, 0.5379]
y = [[2.0000, 0.5379]]
```

## How to think about it

There is no bias and no dropout. The math is three matmuls and an elementwise gate. `x` may have any number of leading dimensions. The gate and up projections read the same `x`, so they can be one matmul against the two weights stacked; that halves the passes over `x` and launches one kernel instead of two. Do not call `torch.nn.functional.silu` for the whole block's job; `torch.sigmoid` or `F.silu` for the gate itself is fine.

## Tests

`D` and `F` from 1 to 512, leading shapes from `[1]` to `[2, 3, 5]`, and large activations where a naive `exp(-z)` overflows. Benchmark input: `x[4, 512, 2048]` with `F = 5632`.

## This version and the other one

This is the PyTorch version of the whole block. The GPU gym's `swiglu` is the CUDA kernel for the elementwise gate alone.
