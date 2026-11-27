# Pad variable-length token batches

Write `solve(sequences, pad_id=0, max_len=None)`.

- `sequences`: a list of token-id lists. The batch may be empty, and any sequence may be empty. Ids are ints.
- `pad_id`: the int written into padding positions. Default `0`. A real token equal to `pad_id` is still real.
- `max_len`: `None`, or an int `≥ 0`. `max_len < 0` raises `ValueError`.

Return `(ids, mask, lengths)`:

- `ids`: an int64 array of shape `(N, W)`.
- `mask`: a boolean array of the same shape. `True` where a real token sits.
- `lengths`: a list of `N` ints, the number of real tokens in each row.

Rules:

- If `max_len` is set, drop tokens from the **right** of each sequence first, keeping `sequence[:max_len]`.
- `W` is the longest remaining length. `max_len` does not add columns past that. If every sequence is empty, `W = 0`.
- Write real tokens on the **left** and `pad_id` on the right. `mask[i, :lengths[i]]` is `True` and the rest is `False`. `lengths[i]` is the length after truncation.
- An empty batch (`N = 0`) returns `ids` of shape `(0, 0)`, `mask` of shape `(0, 0)`, and `lengths = []`, even when `max_len` is set.
- One empty sequence is shape `(1, 0)`, not `(0, 0)`, when nothing else sets a width.

```
solve([[1, 2, 3], [4], [5, 6]])
-> (array([[1, 2, 3],
           [4, 0, 0],
           [5, 6, 0]]),
    array([[ True,  True,  True],
           [ True, False, False],
           [ True,  True, False]]),
    [3, 1, 2])
```

**Interview focus:** Padding ID, validity mask, empty sequences, batch shape.

**Follow-ups** to answer out loud once your solution passes:

- Why pad on the right for training and on the left for batched generation?
- How do bucketing by length or sequence packing reduce wasted compute?
- What goes wrong if the padding mask is missing in attention, or in the loss?
