# Reservoir sampling

Write `solve(stream, k, seed)` that samples `k` items uniformly at random, without replacement, from a stream whose length `n` is unknown in advance.

- `stream`: an **iterator** (a generator). You can loop over it once; it has no `len()` and no indexing. Items are numbers or strings, and equal values may repeat.
- `k`: an integer. Raise `ValueError` if `k < 0`.
- `seed`: an integer. Create `rng = random.Random(seed)` and draw every random number from it, so the same seed gives the same sample.

Return a list of `min(k, n)` items. Each stream **position** may be used at most once, and every position must end up in the sample with the same probability `k / n`. The order of the list does not matter. Memory must be O(k): do not store the stream.

The grader does not expect particular random draws. It checks each sample's validity, that the same seed reproduces the same sample, that every position is kept about `k / n` of the time over thousands of seeds, and the peak memory on a 100,000-item stream. In the test cases a stream is written as a list, or as `{"range": n}` for the integers `0 … n-1`; your function always receives a generator.

```
solve(iter(["a", "b", "c", "d", "e"]), 2, 7)
-> 2 distinct items, for example ["e", "b"]; each letter is included with probability 2/5
```

**Interview focus:** unknown stream length, uniform inclusion probability (and why), O(k) memory.

**Follow-ups** to answer out loud once your solution passes:

- Prove that each item ends up in the sample with probability k/N.
- How would you sample with weights in one pass?
- How would you merge reservoirs from several machines?
