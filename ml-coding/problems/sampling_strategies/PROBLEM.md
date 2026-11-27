# Uniform, weighted, and stratified sampling

Write `solve(n, k, mode, seed, weights, labels)`.

The population is the indices `0 .. n-1`. Return a list of `k` indices, in draw order. Build `rng = random.Random(seed)` and use it for every random number, so the same arguments return the same list. `weights` and `labels` are `None` unless the mode needs them.

- `"uniform_with_replacement"`: each draw is uniform on `0 .. n-1` and independent. `k` may exceed `n`.
- `"uniform_without_replacement"`: `k` distinct indices. Every index has inclusion probability `k / n`. Raise `ValueError` if `k > n`.
- `"weighted"`: `weights` has length `n` and every weight is `>= 0`. Each draw is independent with `P(i) = weight_i / sum(weights)`. A weight of 0 is never drawn. Raise `ValueError` if a weight is negative, `weights` has the wrong length, or `k > 0` and the weights sum to 0. `k` may exceed `n`.
- `"stratified"`: `labels` has length `n`. Sample `k` distinct indices. If class `c` has `n_c` rows, its count in the sample is within ±1 of `k * n_c / n`, between 0 and `n_c`, and the counts sum to `k`. When `k * n_c >= n` the class is large enough to guarantee at least one sample. Raise `ValueError` if `k > n` or `len(labels) != n`.

Also raise `ValueError` if `k < 0`, `n < 0`, `k > 0` and `n == 0`, or `mode` is not one of the four strings. `k == 0` returns `[]`.

```
solve(4, 6, "weighted", 0, [1, 0, 3, 0], None)
-> 6 indices. Index 1 and index 3 never appear. Index 0 has probability 1/4 and index 2 has probability 3/4 on each draw.
```

The grader checks each sample's validity and, over many fixed seeds, the draw rates. It never expects one particular random stream.

**Interview focus:** Replacement, normalized weights, per-class guarantees.

**Follow-ups** to answer out loud once your solution passes:

- When does over-sampling the minority class hurt?
- How do you correct predicted probabilities after training on a resampled dataset?
- How would you sample a fixed fraction of a stream with stratification?
