# Paged attention decode

**Level:** hard   **Teaches:** paged KV caches (vLLM), block tables, gather-based indexing, masking by length

## Task

A serving engine stores the KV cache in fixed-size **blocks** drawn from one shared pool, the way an operating system stores memory in pages. Each sequence has a **block table**: entry `j` names the physical block that holds its tokens `j·S .. j·S + S − 1`, where `S` is the block size. Compute one decode step of attention: each sequence's single query attends to the first `lengths[b]` tokens of its own cache (the new token is already written there).

## Arguments

| Name          | Type           | Shape                 | Role                                              |
| ------------- | -------------- | --------------------- | ------------------------------------------------- |
| `q`           | `torch.Tensor` | `[B, H, Dh]`          | one query per sequence and head, float32, CUDA    |
| `k_pool`      | `torch.Tensor` | `[N, S, H, Dh]`       | the key pool: `N` blocks of `S` token slots       |
| `v_pool`      | `torch.Tensor` | `[N, S, H, Dh]`       | the value pool, same layout                       |
| `block_table` | `torch.Tensor` | `[B, M]`, int64       | physical block ids per sequence; unused entries are 0 |
| `lengths`     | `torch.Tensor` | `[B]`, int64          | tokens in each sequence's cache, `1 <= lengths[b] <= M·S` |
| returns       | `torch.Tensor` | `[B, H, Dh]`          | attention output                                  |

## Formula

```
token t of sequence b lives at  k_pool[block_table[b, t // S], t % S]   (same for v_pool)
scores[b, h, t] = q[b, h] · K_b[t, h] / sqrt(Dh)        for t < lengths[b]
out[b, h]       = softmax over t < lengths[b] of scores[b, h, t], applied to V_b[t, h]
```

## Example

```
S = 2, H = 1, Dh = 1, two sequences, pool of 3 blocks
k_pool = [[[[0]], [[0]]],  [[[1]], [[2]]],  [[[3]], [[9]]]]     (block 0, block 1, block 2)
v_pool = [[[[5]], [[5]]],  [[[10]], [[20]]], [[[30]], [[99]]]]
block_table = [[1, 2], [2, 0]]   lengths = [3, 1]   q = [[[1]], [[1]]]
sequence 0: tokens 0, 1 in block 1, token 2 in block 2 slot 0: keys [1, 2, 3], values [10, 20, 30]
           softmax([1, 2, 3]) = [0.0900, 0.2447, 0.6652]
sequence 1: token 0 in block 2 slot 0: key [3], value [30]
out = [[[25.7521]], [[30.0]]]
```

Slot 1 of block 2 (key 9) is past both sequences' lengths, and block 0 is a padding entry: neither may affect the output.

## How to think about it

Gather each sequence's blocks with advanced indexing, `k_pool[block_table]` is `[B, M, S, H, Dh]`, reshape the block and slot dimensions into one token dimension of length `M·S`, and mask `t >= lengths[b]` with `-inf` before the softmax. Padding entries point at real blocks (block 0) full of other data, so the mask is what keeps them out. A real paged-attention kernel reads the blocks in place instead of gathering a copy.

## Tests

One block per sequence, a length that ends exactly on a block boundary, lengths of 1, two sequences sharing a physical prefix block, blocks in scrambled order, and junk values around `1e4` in every slot past a sequence's length. Benchmark input: `B = 16, H = 16, Dh = 128, S = 16`, up to 128 blocks per sequence.
