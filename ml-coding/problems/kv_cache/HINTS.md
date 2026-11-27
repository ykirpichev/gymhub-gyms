## Hint 1
Keep only the steps that have been written. `get` returns shape `(heads, T, d)` with `T` equal to the number of `prefill` plus `append` steps, not the capacity. A second `prefill`, a too-long prompt, or an append past `capacity` raises.

## Hint 2
One decode step is `append(k, v)` then `attention(query)`. Scores are `einsum("hd,htd->ht", query, keys) / sqrt(d)`, a softmax over time that subtracts the max, then the same weights on the values. Doing this at every position matches causal attention. Attending only to the newest key does not.

## Hint 3
`memory_bytes` counts both tensors. `2 * layers * heads * dim * seq * batch * np.dtype(dtype).itemsize`. `"float16"` is 2 bytes, `"float32"` is 4, `"float64"` is 8.
