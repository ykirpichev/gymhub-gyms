# RMSNorm

**Level:** easy   **Teaches:** reductions over the last dimension, broadcasting, numerical stability

## Task

Normalize each vector along the last dimension of `x` by its root mean square, then scale it by `weight`. Return a new tensor and leave `x` unchanged.

## Arguments

| Name     | Type           | Shape        | Role                   |
| -------- | -------------- | ------------ | ---------------------- |
| `x`      | `torch.Tensor` | `[..., D]`   | input, float32, CUDA   |
| `weight` | `torch.Tensor` | `[D]`        | learned scale          |
| `eps`    | `float`        |              | added inside the root  |
| returns  | `torch.Tensor` | same as `x`  | output                 |

## Formula

```
rms(v) = sqrt(mean(v_i^2) + eps)
y      = x / rms(x) * weight        (rms taken over the last dimension)
```

## Example

```
x      = [[1, 2, 3, 4]]   weight = [1, 1, 1, 1]   eps = 0
mean(x^2) = 7.5,  rms = 2.7386
y      = [[0.3651, 0.7303, 1.0954, 1.4606]]
```

## How to think about it

RMSNorm drops LayerNorm's mean subtraction and bias. It is what Llama-style models use. The only reduction is a mean of squares over `D`, then a broadcasted multiply. Do not call `torch.nn.functional.rms_norm`; build it from tensor operations.

## Tests

Shapes from `[1, 1]` up to `[2, 3, 4, 128]`, plus a case with a large `eps`. Benchmark input: `x[16, 1024, 4096]`.
