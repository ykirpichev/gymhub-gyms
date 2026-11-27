# Append-only KV cache

Write a class `KVCache`. It stores one sequence of keys and values, each of shape `(heads, T, d)`, up to a fixed number of time steps. There is no `solve`. The tests construct `KVCache(capacity)` and replay operations against it.

| Method | Behavior |
| --- | --- |
| `__init__(capacity)` | empty cache. `capacity` is an int `≥ 1`; otherwise raise `ValueError` |
| `prefill(K, V)` | store a prompt. `K` and `V` share shape `(heads, T, d)` with `heads ≥ 1`, `d ≥ 1`, and `1 ≤ T ≤ capacity`. The cache must be empty. Otherwise raise `ValueError` |
| `append(k, v)` | store one new step. `k` and `v` share shape `(heads, d)`. On an empty cache this sets `heads` and `d`. Later steps must match, and `T + 1` must not exceed `capacity`. Otherwise raise `ValueError` |
| `get()` | `(K, V)` for the filled prefix only, each a float64 array of shape `(heads, T, d)` in time order. Raise `ValueError` when the cache is empty |
| `attention(query)` | `query` has shape `(heads, d)`, one vector per head. Raise `ValueError` when the cache is empty or the shape does not match. Return a float64 array of shape `(heads, d)` |
| `memory_bytes(layers, heads, dim, seq, batch, dtype)` | static. Bytes for a stack of key and value caches |

`attention` is scaled dot-product attention over the **filled** prefix, independently per head:

- scores = `(query · keys) / sqrt(d)`, over the `T` cached steps
- weights = a max-subtracted softmax over those `T` scores
- output = weights times the cached values

Scores can be as large as `1e6`. The softmax must stay finite.

Appending step `t` and then calling `attention` with that step's query matches full causal attention on the same keys and values: position `t` attends only to steps `0..t`.

`memory_bytes` returns the Python int

`2 * layers * heads * dim * seq * batch * itemsize`

where `itemsize` is `np.dtype(dtype).itemsize` (`"float16"`, `"float32"`, `"float64"`, and any other NumPy dtype name). `layers`, `heads`, `dim`, `seq`, and `batch` are non-negative ints. An unknown dtype raises `ValueError`.

```
cache = KVCache(4)
cache.prefill(K=[[[1, 0, 0, 0], [0, 1, 0, 0]]],
               V=[[[1, 0, 0, 0], [0, 0, 1, 0]]])
cache.attention([[1, 0, 0, 0]])
-> array([[0.6225, 0.    , 0.3775, 0.    ]])
cache.append([[0, 0, 1, 0]], [[0, 0, 0, 4]])
cache.attention([[0, 0, 1, 0]])
-> array([[0.2741, 0.    , 0.2741, 1.8075]])
KVCache.memory_bytes(1, 1, 4, 8, 1, "float32")
-> 256
```

`d = 4`, so the scale is `1/2`. The first query dots the keys as `[1, 0]`, and the scores are `[0.5, 0]`. After the append the second query scores `[0, 0, 0.5]` against the three cached keys. `get()` then has shape `(1, 3, 4)`.

**Interview focus:** Cache shape, capacity, prefill versus decode, memory growth.

## This version and the other one

This is the CPU interview version: one sequence, an append-only class, and the validation an interviewer asks about. The LLM gym's `kv_cache_decode` is the GPU version: one decode step for a batch of sequences of different lengths in a padded cache, benchmarked.

**Follow-ups** to answer out loud once your solution passes:

- How many bytes per token does the cache take for your model's shapes, and how does grouped-query attention change that?
- How does a paged cache avoid reserving memory for the maximum length?
- Two requests share a long system prompt. How could they share cache?
