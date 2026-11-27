# Indexing in one page

**Basic indexing** (integers and slices) returns a view. **Advanced indexing** (integer arrays, boolean masks) returns a copy, and its shape follows the index arrays:

```python
x = np.arange(20).reshape(4, 5)
x[[0, 2, 3]]                    # rows 0, 2, 3: shape (3, 5)
x[[0, 2], [1, 4]]               # elements (0, 1) and (2, 4): shape (2,)
x[np.arange(4), [4, 0, 1, 2]]   # one element per row: shape (4,)
x[x % 2 == 0]                   # a mask flattens: shape (10,)
```

Picking one element per row, at a column given per row, is everywhere: the log-probability of the right class in a loss, the score of a chosen token. Three ways to write it: `x[np.arange(n), idx]`, `np.take_along_axis(x, idx[:, None], axis=1)`, and in PyTorch `x.gather(1, idx.unsqueeze(1))`.

The reverse, **scatter**, writes or accumulates values at indices. Beware repeated indices:

```python
out = np.zeros(3)
out[[0, 0, 2]] += 1          # out == [1, 0, 1]: the repeated 0 is written once, not added twice
np.add.at(out, [0, 0, 2], 1) # out == [2, 0, 1]: unbuffered, accumulates
```

In PyTorch the accumulating versions are `index_add_`, `scatter_add_` and `scatter_reduce_`, and `torch.bincount` counts. A segment sum (sum the rows of each group) is a scatter-add; so is the backward of an embedding lookup, and so is combining the outputs of a mixture-of-experts layer.

> [!WARNING]
> On a GPU, scatter-add uses atomics, so the order of additions varies and floating-point results can differ slightly between runs. `torch.use_deterministic_algorithms(True)` makes such operations deterministic or raises an error.
