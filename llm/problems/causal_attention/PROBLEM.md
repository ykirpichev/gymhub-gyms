# Causal self-attention

**Level:** medium   **Teaches:** batched matmul, masking, stable softmax, the memory cost of the score matrix

## Task

Compute scaled dot-product attention for a decoder: every query position `i` attends to key positions `j <= i` only (it cannot see the future). Batch and head dimensions are independent. Return a new tensor.

## Arguments

| Name    | Type           | Shape           | Role                   |
| ------- | -------------- | --------------- | ---------------------- |
| `q`     | `torch.Tensor` | `[B, H, T, Dh]` | queries, float32, CUDA |
| `k`     | `torch.Tensor` | `[B, H, T, Dh]` | keys                   |
| `v`     | `torch.Tensor` | `[B, H, T, Dh]` | values                 |
| returns | `torch.Tensor` | `[B, H, T, Dh]` | attention output       |

## Formula

```
S[i, j] = (q_i · k_j) / sqrt(Dh)        if j <= i
S[i, j] = -inf                          if j >  i
P[i, :] = softmax(S[i, :])              (over j)
out_i   = sum_j P[i, j] * v_j
```

## Example

```
B = H = 1, T = 2, Dh = 2
q = [[0, 0], [0, 2]]   k = [[1, 0], [1, 1]]   v = [[1, 2], [3, 4]]
row 0: sees only j = 0              -> out_0 = v_0 = [1, 2]
row 1: S = [0, 2/sqrt(2)] = [0, 1.4142]
       P = [0.1956, 0.8044]
       out_1 = 0.1956·[1, 2] + 0.8044·[3, 4] = [2.6089, 3.6089]
```

## How to think about it

`q @ k.transpose(-2, -1)` gives all `T × T` scores per head at once. Apply the mask before the softmax, by filling the upper triangle with `-inf`; each row keeps at least its diagonal entry, so no row is all `-inf`. Softmax must subtract the row max to stay finite on large scores. For the benchmark the score matrix is `4·16·1024·1024` floats = 256 MiB, so materializing it is a real cost; flash-attention style kernels avoid it by streaming over key blocks with an online softmax. Do not call `torch.nn.functional.scaled_dot_product_attention` or `torch.nn.MultiheadAttention`.

## Tests

`T` from 1 to 256 (including odd `T = 129`), `Dh` 4 to 128, and inputs with large score magnitudes. Tolerance `1e-4`. Benchmark input: `B = 4, H = 16, T = 1024, Dh = 64`.

## This version and the other one

This is the GPU version: one batched causal pass over CUDA tensors in PyTorch, with a benchmark that charges you for the `L × L` score matrix. ML coding's `causal_attention_mask` and `scaled_dot_product_attention` are the CPU interview versions in NumPy: they build the padding-aware mask and the masked softmax step by step, with no benchmark.
