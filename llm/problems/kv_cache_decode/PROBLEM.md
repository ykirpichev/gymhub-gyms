# KV-cache decode step

**Level:** hard   **Teaches:** padded batched caches, per-sequence masks, scatter by index, decode-time memory bandwidth

## Task

During generation a model keeps the keys and values of past tokens in a preallocated cache of capacity `L`. The batch holds sequences of different lengths, so sequence `b` has only its first `lengths[b]` slots filled; everything at or after that index is stale garbage. Now one new token arrives per sequence, with query `q_new`, key `k_new` and value `v_new`.

The new token occupies position `lengths[b]`. Its query attends to cache positions `0 .. lengths[b] - 1` and to itself (the new key/value), and to nothing else. Return the attention output for the new token. You may update the cache or not, as you like; only the returned tensor is checked.

## Arguments

| Name      | Type           | Shape           | Role                                             |
| --------- | -------------- | --------------- | ------------------------------------------------ |
| `k_cache` | `torch.Tensor` | `[B, H, L, Dh]` | cached keys, float32, CUDA                       |
| `v_cache` | `torch.Tensor` | `[B, H, L, Dh]` | cached values                                    |
| `lengths` | `torch.Tensor` | `[B]`           | int64, valid prefix length per sequence, `< L`   |
| `q_new`   | `torch.Tensor` | `[B, H, 1, Dh]` | new token's query                                |
| `k_new`   | `torch.Tensor` | `[B, H, 1, Dh]` | new token's key                                  |
| `v_new`   | `torch.Tensor` | `[B, H, 1, Dh]` | new token's value                                |
| returns   | `torch.Tensor` | `[B, H, 1, Dh]` | attention output of the new token                |

## Formula

For each `b, h`, with `n = lengths[b]`:

```
keys    K = [k_cache[b, h, 0], ..., k_cache[b, h, n-1], k_new[b, h, 0]]     (n + 1 rows)
values  V = [v_cache[b, h, 0], ..., v_cache[b, h, n-1], v_new[b, h, 0]]
s_j     = (q_new[b, h, 0] · K_j) / sqrt(Dh)
p       = softmax(s)
out[b, h, 0] = sum_j p_j V_j
```

Slots `n .. L-1` of the cache never contribute, whatever they contain.

## Example

```
B = H = 1, L = 3, Dh = 2, lengths = [1]
k_cache = [[1, 0], [9, 9], [9, 9]]     v_cache = [[1, 2], [9, 9], [9, 9]]   (slots 1, 2 are stale)
q_new = [0, 2]   k_new = [1, 1]   v_new = [3, 4]
K = [[1, 0], [1, 1]]   V = [[1, 2], [3, 4]]
s = [0, 2/sqrt(2)] = [0, 1.4142]   p = [0.1956, 0.8044]
out = [[[[2.6089, 3.6089]]]]
```

## How to think about it

A loop over `b` with Python slicing is correct but launches many tiny kernels. Instead, express the per-sequence lengths as a `[B, L]` boolean mask built from `torch.arange(L)` and `lengths`, and do the whole batch with one matmul. There are two ways to handle the new token: write it into (a copy of) the cache at index `lengths[b]` and mask `j > lengths[b]`, or keep it separate, mask `j >= lengths[b]` on the cache scores and append its own score as an extra column. The second avoids copying the cache. Stale slots hold large values, so an unmasked or off-by-one solution fails loudly. Decode is memory-bound: the cost is reading `k_cache` and `v_cache` once, so every extra full copy shows up in the benchmark.

## Tests

Empty caches (`lengths = 0`, the output is `v_new`), full caches (`lengths = L - 1`), mixed lengths inside one batch, `Dh` up to 128, and stale slots holding values around `1e4`. Benchmark input: `B = 8, H = 32, L = 4096, Dh = 128` (512 MiB per cache) with random lengths.

## This version and the other one

This is the GPU version: one decode step for a batch of sequences of different lengths in a padded, preallocated cache, benchmarked for memory bandwidth. ML coding's `kv_cache` is the CPU interview version: a single-sequence, append-only cache class with its capacity checks.
