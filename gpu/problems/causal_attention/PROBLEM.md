# Causal flash attention

**Level:** hard   **Teaches:** FlashAttention with a causal mask, skipping work, one program per (head, query tile)

## Task

Compute self-attention for `H` heads where each token may only attend to itself and earlier tokens. This is what a decoder-only language model runs in every layer during training and prefill. It is `torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)` on float32.

Do `attention` first. This exercise adds heads and a causal mask, and the benchmark rewards a kernel that never touches the masked half.

## Arguments

| Name | Type           | Shape       | Role   |
| ---- | -------------- | ----------- | ------ |
| `Q`  | `const float*` | `H × L × d` | input  |
| `K`  | `const float*` | `H × L × d` | input  |
| `V`  | `const float*` | `H × L × d` | input  |
| `O`  | `float*`       | `H × L × d` | output |
| `H`  | `int`          |             | size   |
| `L`  | `int`          |             | size   |
| `d`  | `int`          |             | size   |

Row `i` of head `h` starts at `Q[(h * L + i) * d]`, and the same for `K`, `V` and `O`. `d` is at most 128.

## Formula

For each head separately:

```
S[i, j] = (sum over k of Q[i, k] * K[j, k]) / sqrt(d)      for j <= i
S[i, j] = -infinity                                         for j > i
P[i, :] = softmax(S[i, :])
O[i, k] = sum over j of P[i, j] * V[j, k]
```

Row 0 attends only to itself, so `O[0] = V[0]`.

## Example

```
H = 1, L = 2, d = 2
Q = [[1, 0],     K = [[1, 0],     V = [[1, 2],
     [0, 2]]          [0, 1]]          [3, 4]]

row 0 sees key 0 only:         O[0] = V[0] = [1, 2]
row 1 sees keys 0 and 1:       S = [0, 1.4142]    P = [0.1956, 0.8044]
                               O[1] = 0.1956 * [1, 2] + 0.8044 * [3, 4] = [2.6089, 3.6089]
```

## How to think about it

Start from your `attention` kernel. Each program (or block) owns one head and one tile of `BLOCK_N` query rows, and walks over the keys in tiles of `BLOCK_M`, keeping a running max, a running sum and an output accumulator per query row.

The mask changes two things. Inside a tile that straddles the diagonal, scores with `j > i` must become `-inf` before the softmax update. And every key tile that lies entirely after the query tile contributes nothing, so do not load it at all: the loop over key tiles for query tile `q` can stop at the diagonal. That halves the work, and the benchmark counts only the unmasked half, so a kernel that computes everything and masks it scores half the TFLOP/s.

Query tiles near the end of the sequence do more work than tiles near the start. With many heads there are enough programs to keep the GPU busy anyway, but think about which order they are launched in.

## In an interview

Flash attention with a causal mask: block skipping and the diagonal tiles are the details interviewers probe.

Follow-ups to answer out loud once your kernel passes:

- Which key blocks can a query block skip, and how much work does that save?
- How do you mask inside the diagonal tile, and what must a fully masked row produce?
- How does decoding with a KV cache change the shape of this problem?

## This version and the other one

This is the kernel version: you write the tiled loop yourself in CUDA or Triton and are benchmarked in TFLOP/s. The LLM gym's `causal_attention` is the PyTorch version over batched tensors with padding, where the point is to get the mask right and to avoid materializing more than you need.
