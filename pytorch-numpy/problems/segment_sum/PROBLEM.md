# Segment sums

Write `solve(values, segment_ids, num_segments)`:

- `values`: an `N × D` array of floats.
- `segment_ids`: `N` ints in `0 .. num_segments − 1`, in any order and with repeats.

Return a tuple `(sums, means)`: `sums[s]` is the sum of the rows with segment id `s` (zeros for an empty segment), and `means[s]` is their mean (zeros for an empty segment). Both are `num_segments × D` NumPy arrays. No Python loop over rows.

```
solve([[1, 2], [3, 4], [5, 6]], [1, 0, 1], 3)
-> (array([[3, 4], [6, 8], [0, 0]]), array([[3, 4], [3, 4], [0, 0]]))
```

#### What interviewers look for

- `np.add.at` (or `index_add_` in PyTorch), not `sums[ids] += values`.
- Guarding the division for empty segments.

#### Follow-ups

- Where does this operation appear in ML? (Embedding backward, graph neural network message passing, MoE combine, bag-of-words.)
- How would you compute a per-segment max? A per-segment softmax?
- Why might the GPU result differ in the last bits between runs?
