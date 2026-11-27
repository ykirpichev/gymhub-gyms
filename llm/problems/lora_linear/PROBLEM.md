# LoRA linear layer

**Level:** easy   **Teaches:** matmul shapes, associativity and its cost, low-rank adapters

## Task

LoRA fine-tunes a model by freezing a linear layer `W` and learning a small update `ΔW = B A` of rank `r`. Compute the forward pass of such a layer: the frozen projection plus bias, plus the low-rank path scaled by `alpha / r`. Return a new tensor.

## Arguments

| Name    | Type           | Shape          | Role                                  |
| ------- | -------------- | -------------- | ------------------------------------- |
| `x`     | `torch.Tensor` | `[N, in]`      | input rows, float32, CUDA             |
| `W`     | `torch.Tensor` | `[out, in]`    | frozen weight                         |
| `b`     | `torch.Tensor` | `[out]`        | frozen bias                           |
| `A`     | `torch.Tensor` | `[r, in]`      | adapter down-projection               |
| `B`     | `torch.Tensor` | `[out, r]`     | adapter up-projection                 |
| `alpha` | `float`        |                | adapter scale numerator               |
| returns | `torch.Tensor` | `[N, out]`     | output                                |

`r` is `A.shape[0]`.

## Formula

```
y = x W^T + b + (alpha / r) * (x A^T) B^T
```

Mathematically this equals `x (W + (alpha / r) B A)^T + b`.

## Example

```
x = [[1, 2]]
W = [[1, 0], [0, 1], [1, 1]]   b = [0, 0, 1]
A = [[1, 1]]                   B = [[1], [0], [-1]]   alpha = 2, r = 1
x W^T + b   = [1, 2, 4]
x A^T       = [3]
(x A^T) B^T = [3, 0, -3],  scaled by 2 -> [6, 0, -6]
y           = [[7, 2, -2]]
```

## How to think about it

The parenthesization decides the cost. `(x A^T) B^T` is two thin matmuls costing about `N·r·(in + out)` multiply-adds. Forming `B A` first builds a full `[out, in]` matrix: `out·in·r` work plus a big temporary, and if you then add it to `W` you copy all of `W` every call. With `r = 16` and `in = out = 4096`, the thin path is cheap next to the main `x W^T`. Merging `B A` into `W` only pays off when you do it once and reuse the merged weight for many calls (inference), not per forward pass.

## Tests

Sizes from all-ones up to `[33, 300] -> 700` with `r = 16`, a zero `B` (an adapter at initialization), and `r` equal to the layer width. Benchmark input: `x[4096, 4096]`, `W[4096, 4096]`, `r = 16`.

## This version and the other one

This is the GPU version: the forward pass only, on CUDA tensors, with a benchmark, so the order of the matrix products shows up in the time. ML coding's `lora_linear` is the CPU interview version in NumPy: it also asks for the merged weight and the parameter counts.
