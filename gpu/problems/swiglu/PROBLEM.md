# Fused SwiGLU

**Level:** easy   **Teaches:** kernel fusion, 2D indexing into a packed matrix, a safe `exp`

## Task

The feed-forward block of Llama-style models computes a gate and an "up" projection with one matrix multiply, so they arrive packed side by side in one matrix. Each row of `X` holds `cols` gate values followed by `cols` up values. Write `silu(gate) * up` to `Y`.

## Arguments

| Name   | Type           | Shape             | Role   |
| ------ | -------------- | ----------------- | ------ |
| `X`    | `const float*` | `rows × (2·cols)` | input  |
| `Y`    | `float*`       | `rows × cols`     | output |
| `rows` | `int`          |                   | size   |
| `cols` | `int`          |                   | size   |

Row-major: the gate for `(r, c)` is `X[r * 2 * cols + c]` and the up value is `X[r * 2 * cols + cols + c]`.

## Formula

```
silu(g) = g / (1 + exp(-g))
Y[r, c] = silu(X[r, c]) * X[r, cols + c]
```

## Example

```
X = [[ 0, 1,   2, 3],          gate = [[ 0, 1],    up = [[2, 3],
     [-1, 2, 0.5, 4]]                  [-1, 2]]          [0.5, 4]]

silu(gate) = [[ 0,       0.7311],
              [-0.2689,  1.7616]]
Y          = [[ 0,       2.1932],
              [-0.1345,  7.0464]]
```

## How to think about it

Written in PyTorch as `F.silu(gate) * up`, this is two kernels: the first writes `silu(gate)` to memory and the second reads it back. A fused kernel reads each input once and writes each output once, which is all a memory-bound operation can do. That is the whole point of the exercise: count the bytes.

Use one thread per output element. The only trap is `exp(-g)` for very negative `g`: it becomes `inf`, and `g / inf` is `-0`, which is the right answer. Make sure your formula does not turn that into `NaN`.

## In an interview

Tests fusion: gate and up projections, SiLU and a product, done in one memory pass instead of three.

Follow-ups to answer out loud once your kernel passes:

- Count the bytes per element fused versus unfused.
- Why do LLM MLPs use gated units with a hidden size of about 8/3 of the model width?
- Could you fuse this into the matmul that produces the two inputs? What would the epilogue look like?
