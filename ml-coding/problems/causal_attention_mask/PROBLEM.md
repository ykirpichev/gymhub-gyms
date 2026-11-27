# Build a causal attention mask

Write `solve(valid)` for a batch of `B` padded token sequences of length `L`.

- `valid`: a `B × L` list of lists of bools (or bool array). `valid[b][t]` is True when position `t` of sequence `b` holds a real token and False when it is padding. Padding can be anywhere (right-padded, left-padded, or holes).

Return a NumPy **bool** array `mask` of shape `(B, L, L)`, indexed `mask[b, i, j]` with `i` the **query** position and `j` the **key** position:

```
mask[b, i, j] = (j <= i) and valid[b][j] and valid[b][i]
```

True means "query `i` may attend to key `j`". A query attends to itself and to earlier real tokens only. A padding query attends to nothing (its row is all False), and a padding key is never attended to (its column is all False).

`L = 0` gives shape `(B, 0, 0)`. An empty batch `valid = []` gives shape `(0, 0, 0)`.

```
solve([[True, True, True], [True, True, False]])
-> array([[[ True, False, False],
           [ True,  True, False],
           [ True,  True,  True]],
          [[ True, False, False],
           [ True,  True, False],
           [False, False, False]]])
```

**Interview focus:** query/key axis semantics (rows are queries), `j <= i` rather than `j < i`, combining causal and padding masks by broadcasting, and the True-means-attend convention.

## This version and the other one

This is the CPU interview version in NumPy, and it is only the mask: padding anywhere in the row, and the exact True/False layout. The LLM gym's `causal_attention` is the GPU version: one batched causal attention pass over CUDA tensors, benchmarked.

**Follow-ups** to answer out loud once your solution passes:

- How do you combine a causal mask with a padding mask, and what dtype is cheapest?
- What does a sliding-window mask look like, and why does it cap the KV cache?
- How does a prefix-LM mask differ from a causal one?
