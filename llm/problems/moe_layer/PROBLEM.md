# Mixture-of-experts layer

**Level:** hard   **Teaches:** top-k routing, grouping tokens by expert, gather and scatter-add, sparse compute

## Task

Implement a Mixtral-style sparse MoE feed-forward layer. A router scores every expert for every token; each token goes to its `top_k` best experts, and its output is the gate-weighted sum of those experts' outputs. Every expert is a SwiGLU MLP.

## Arguments

| Name       | Type           | Shape        | Role                                   |
| ---------- | -------------- | ------------ | -------------------------------------- |
| `x`        | `torch.Tensor` | `[T, D]`     | tokens, float32, CUDA                  |
| `router_w` | `torch.Tensor` | `[E, D]`     | router weight                          |
| `w_gate`   | `torch.Tensor` | `[E, F, D]`  | each expert's gate projection          |
| `w_up`     | `torch.Tensor` | `[E, F, D]`  | each expert's up projection            |
| `w_down`   | `torch.Tensor` | `[E, D, F]`  | each expert's down projection          |
| `top_k`    | `int`          |              | experts per token, `1 <= top_k <= E`   |
| returns    | `torch.Tensor` | `[T, D]`     | output                                 |

## Formula

```
logits      = x @ router_w^T                          # [T, E]
idx, val    = top_k largest logits per token          # [T, k]
gates       = softmax(val) over the k chosen experts  # renormalised over the chosen ones only
expert_e(v) = (silu(v @ w_gate[e]^T) * (v @ w_up[e]^T)) @ w_down[e]^T
y[t]        = sum over j of gates[t, j] * expert_{idx[t, j]}(x[t])
```

## Example

```
x = [[1, 0]]   router_w = [[2, 0], [1, 0], [0, 0]]   top_k = 2
logits = [2, 1, 0]: experts 0 and 1 are chosen, gates = softmax([2, 1]) = [0.7311, 0.2689]
y = 0.7311 * expert_0(x) + 0.2689 * expert_1(x)
```

The test inputs are random, so two router logits of one token are never tied.

## How to think about it

Running every expert on every token and masking is correct, but does `E / top_k` times the necessary work. Instead, flatten the `T × k` assignments, sort or group them by expert, and for each expert gather its tokens into one dense batch, run its MLP, scale by the gates, and scatter-add the results back with `index_add_`. A Python loop over the `E` experts is fine (8 to 64 iterations); a loop over tokens is not.

## Tests

`top_k = 1` and `top_k = E`, experts that receive no tokens, a single token, and random shapes up to `T = 256`, `E = 8`. Benchmark input: `T = 4096`, `D = 1024`, `F = 2048`, `E = 8`, `top_k = 2`.

## This version and the other one

This is the PyTorch version of the whole layer, routing included. The GPU gym's `moe_matmul` is the CUDA kernel for the grouped matmul alone, with the routing given.
