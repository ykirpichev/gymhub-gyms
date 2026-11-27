# Grouped-query attention

**Level:** medium   **Teaches:** grouped-query and multi-query attention, broadcasting instead of copying, KV-cache size

## Task

Compute causal self-attention where there are fewer key/value heads than query heads. Query head `h` uses key/value head `h // G`, where `G = Hq / Hkv` is the group size. `Hkv = Hq` is ordinary multi-head attention; `Hkv = 1` is multi-query attention.

## Arguments

| Name    | Type           | Shape               | Role                       |
| ------- | -------------- | ------------------- | -------------------------- |
| `q`     | `torch.Tensor` | `[B, Hq, L, Dh]`    | queries, float32, CUDA     |
| `k`     | `torch.Tensor` | `[B, Hkv, L, Dh]`   | keys                       |
| `v`     | `torch.Tensor` | `[B, Hkv, L, Dh]`   | values                     |
| returns | `torch.Tensor` | `[B, Hq, L, Dh]`    | output                     |

`Hq` is a multiple of `Hkv`.

## Formula

```
g          = h // (Hq / Hkv)
scores     = q[b, h] @ k[b, g]^T / sqrt(Dh)          # [L, L]
scores[i, j] = -inf  where j > i                    # causal
out[b, h]  = softmax(scores, dim=-1) @ v[b, g]
```

## Example

```
B = 1, Hq = 2, Hkv = 1, L = 2, Dh = 1
q = [[[[1], [1]], [[2], [0]]]]   k = [[[[0], [1]]]]   v = [[[[10], [20]]]]
head 0, position 1: scores [0, 1] -> weights [0.2689, 0.7311] -> 17.311
head 1, position 1: scores [0, 0] -> weights [0.5, 0.5]       -> 15.0
position 0 always sees only key 0: output 10
out = [[[[10], [17.3106]], [[10], [15.0]]]]
```

## How to think about it

`k.repeat_interleave(G, dim=1)` makes the shapes match and is correct, but it copies every KV head `G` times, which is exactly the memory GQA exists to save. Instead, view the queries as `[B, Hkv, G, L, Dh]` and the keys as `[B, Hkv, 1, L, Dh]`: matmul broadcasts the singleton group dimension without a copy. Do not call `scaled_dot_product_attention`.

## Tests

MHA (`Hkv = Hq`), MQA (`Hkv = 1`), `G` from 2 to 8, `L = 1`, odd `L`, and `Dh` up to 128. Benchmark input: `B = 4, Hq = 32, Hkv = 8, L = 1024, Dh = 128`.
