# Scaled dot-product attention

Write `solve(q, k, v, mask=None)`.

- `q`: queries, shape `(..., Lq, d_k)`.
- `k`: keys, shape `(..., Lk, d_k)`.
- `v`: values, shape `(..., Lk, d_v)`.
- `mask`: `None`, or a boolean array that broadcasts to `(..., Lq, Lk)`. `mask[..., i, j]` is **True when query `i` may attend to key `j`**.

The leading `...` batch dimensions (zero or more) are the same for `q`, `k` and `v`. Inputs may be nested lists; work in float64.

Return a tuple `(output, weights)`:

- `weights`, shape `(..., Lq, Lk)`: `softmax(q kᵀ / sqrt(d_k))` over the last axis, where positions with `mask == False` get weight exactly 0 and the remaining weights of the row sum to 1.
- `output`, shape `(..., Lq, d_v)`: `weights @ v`.

Rules:

- Apply the mask to the scores **before** the softmax (masked scores act as `-inf`).
- Scores can be as large as `1e6`; the softmax must stay finite (no `inf`, no `nan`).
- If every key is masked for a query, that query's weights are all 0 and its output row is all 0.

```
solve(q=[[1, 0], [0, 1]], k=[[1, 0], [0, 1]], v=[[1, 2], [3, 4]],
      mask=[[True, False], [True, True]])
-> (array([[1.    , 2.    ],
           [2.3395, 3.3395]]),
    array([[1.    , 0.    ],
           [0.3302, 0.6698]]))
```

Query 1 scores both keys: `[0, 1] / sqrt(2)`, so its weights are `softmax([0, 0.7071]) = [0.3302, 0.6698]`.

**Interview focus:** Q/K/V shapes and batched matmuls, the `1/sqrt(d_k)` scale, masking before a max-subtracted softmax, and fully masked rows.

## This version and the other one

This is the CPU interview version in NumPy: any leading batch shape, an explicit boolean mask, and both the output and the weights returned. The LLM gym's `causal_attention` is the GPU version: causal only, batched CUDA tensors, benchmarked, and it returns just the output.

**Follow-ups** to answer out loud once your solution passes:

- Why divide by sqrt(d_k)?
- What is the memory of the score matrix at 32K tokens, and how does flash attention avoid it?
- How does multi-head attention reshape the tensors?
