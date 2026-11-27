#### The idea
A scatter-add: row `i` is added into `sums[segment_ids[i]]`. The trap is repeated ids. Fancy-index assignment
`sums[ids] += x` reads all targets first and writes each once, so duplicates are lost. `np.add.at` is the unbuffered
version that applies every addition.

#### A solution
```python
import numpy as np


def solve(values, segment_ids, num_segments):
    x = np.asarray(values, dtype=np.float64)
    ids = np.asarray(segment_ids, dtype=np.int64)
    sums = np.zeros((num_segments, x.shape[1]))
    np.add.at(sums, ids, x)
    counts = np.bincount(ids, minlength=num_segments).astype(np.float64)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    return (sums, means)
```

#### In PyTorch and on a GPU
`sums.index_add_(0, ids, x)` or `scatter_add_` / `scatter_reduce_`. On a GPU these use atomics, so the order of
additions varies and float results can differ run to run; sorting by segment and doing a segmented reduction is the
deterministic alternative. This pattern is everywhere: embedding-bag, graph neural networks (summing messages per
node) and MoE (combining expert outputs per token).

#### Pitfalls
- `sums[ids] += x`: silently wrong when an id repeats.
- Dividing by zero for empty segments; `np.divide(..., where=...)` keeps them at zero.
