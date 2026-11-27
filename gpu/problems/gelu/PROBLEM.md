# GELU activation

**Level:** easy   **Teaches:** elementwise math, `expf` and `tanhf`, `float4` vectorization

## Task

Apply the GELU activation to every element of `X` and write the result to `Y`. Use the tanh approximation, the version used in GPT-2 and many other models.

## Arguments

| Name | Type           | Shape | Role   |
| ---- | -------------- | ----- | ------ |
| `X`  | `const float*` | `N`   | input  |
| `Y`  | `float*`       | `N`   | output |
| `N`  | `int`          |       | size   |

## Formula

```
u    = sqrt(2 / pi) * (x + 0.044715 * x^3)
Y[i] = 0.5 * x * (1 + tanh(u))          where x = X[i]
```

`sqrt(2 / pi)` is about `0.7978845608`. This matches `torch.nn.functional.gelu(x, approximate="tanh")`.

## Example

```
X         = [-2,      -1,      0,  1,      2     ]
u         = [-1.8812, -0.8336, 0,  0.8336, 1.8812]
tanh(u)   = [-0.9546, -0.6824, 0,  0.6824, 0.9546]
Y         = [-0.0454, -0.1588, 0,  0.8412, 1.9546]
```

## How to think about it

Each output depends only on the input at the same index. One thread per element is correct and simple.

The math is a few multiplies and one `tanh`. That is cheap next to reading and writing 8 bytes per element from memory, so the kernel is memory-bound. A fast version spends little time per byte: it issues wide loads, keeps the arithmetic in float32 (`tanhf`, not `tanh`), and does not repeat work. Large positive inputs give `tanh(u)` close to 1 and `Y` close to `x`. Large negative inputs give `Y` close to 0. Make sure your formula does not produce `inf` or `NaN` at either end.

## In an interview

Tests whether you know elementwise kernels are memory-bound even with `tanhf` or `erff` in them, and about fast math.

Follow-ups to answer out loud once your kernel passes:

- The exact GELU uses `erf`, the approximation `tanh`. Which would you ship, and how do you check the error?
- What does `--use_fast_math` change, and when is it unsafe?
- Fuse GELU with the bias add of the matmul before it. What is the speedup if both were memory-bound?
