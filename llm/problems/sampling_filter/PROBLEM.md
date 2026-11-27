# Top-k / top-p filtering

**Level:** medium   **Teaches:** topk and sort, cumulative sums, scatter back to the original order, masking with -inf

## Task

Before sampling the next token, generation code usually throws away unlikely tokens. Implement the two standard filters on a batch of logit rows and return the filtered logits: removed entries become `-inf`, kept entries keep their exact original value. Each row is filtered independently.

1. **Top-k** (skipped when `top_k == 0` or `top_k >= V`): keep the `top_k` largest logits of the row.
2. **Top-p / nucleus** (skipped when `top_p == 1.0`): take the softmax of the logits left after step 1 (removed entries get probability 0). Sort tokens by probability, highest first, and keep the smallest prefix whose total probability is `>= top_p`. The most likely token is always kept.

The order matters: top-p is computed on the distribution renormalized after top-k.

## Arguments

| Name     | Type           | Shape    | Role                                            |
| -------- | -------------- | -------- | ----------------------------------------------- |
| `logits` | `torch.Tensor` | `[B, V]` | float32, CUDA; values in a row are all distinct |
| `top_k`  | `int`          |          | `0` disables                                    |
| `top_p`  | `float`        |          | in `(0, 1]`; `1.0` disables                     |
| returns  | `torch.Tensor` | `[B, V]` | filtered logits                                 |

## Formula

```
after top-k:  x[j] = logits[j]   if logits[j] is among the top_k largest, else -inf
p        = softmax(x)
before[j] = sum of p[i] over tokens i with p[i] > p[j]      (mass strictly ahead of j)
keep j for top-p  iff  before[j] < top_p
```

`before` is 0 for the most likely token, so it is always kept. A token whose `before` is within `1e-5` of `top_p` is accepted either way, since rounding can push it across.

## Example

```
logits = [[2, 1, 0, -1, 3]]   top_k = 4   top_p = 0.8
top-k drops -1:        x = [2, 1, 0, -inf, 3]
softmax:               p = [0.2369, 0.0871, 0.0321, 0, 0.6439]
sorted by p:   3 (0.6439)  2 (0.2369)  1 (0.0871)  0 (0.0321)
before:            0         0.6439      0.8808      0.9679
keep before < 0.8: the tokens with logit 3 and 2
result = [[2, -inf, -inf, -inf, 3]]
```

## How to think about it

`torch.topk(..., k).values[:, -1:]` is the per-row threshold for top-k. For top-p, sort the row descending, take the softmax, and use `cumsum`: the mass strictly before position `r` is `cumsum[r] - p[r]`. That gives a removal mask in sorted order; `scatter` puts it back in vocabulary order. Note that `softmax` is invariant to the order, so you can sort logits and take softmax of the sorted row. With large vocabularies, sorting the full row is the expensive part; when top-k is active, only `top_k` values can survive, so the sort can be much smaller.

## Tests

Both filters disabled, each alone, both together, `top_k = 1`, `top_k > V`, and a tiny `top_p` that keeps only the best token. Benchmark input: `logits[64, 128256]`, `top_k = 1000`, `top_p = 0.95`.

## This version and the other one

This is the GPU version: it filters a batch of logit rows on CUDA and returns filtered logits of the same shape (removed entries `-inf`); drawing the token is left to the caller. ML coding's `top_k_top_p_sampling` is the CPU interview version for one row: it also applies temperature and draws the token, so it returns `(probs, token)`.
