#### The idea
Compute every test-to-train squared distance at once with the expansion
`‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b`. That is one matrix product `B @ A.T` plus two broadcast vectors, and it never builds
the `M × N × D` difference tensor. Square roots are not needed: they do not change the order.

Then take the `k` smallest per row. A **stable** argsort keeps the lower training index first among equal distances,
which is exactly the tie rule. `np.bincount(...).argmax()` returns the first (smallest) label with the highest count,
which is the vote rule.

#### A solution
```python
import numpy as np


def solve(X_train, y_train, X_test, k):
    A = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    B = np.asarray(X_test, dtype=np.float64)
    if not 1 <= k <= len(A):
        raise ValueError("k must be between 1 and the number of training points")
    # Squared distances, M x N, without an M x N x D intermediate.
    d2 = (B ** 2).sum(axis=1)[:, None] + (A ** 2).sum(axis=1)[None, :] - 2.0 * B @ A.T
    # A stable sort keeps the lower training index first among equal distances.
    nearest = np.argsort(d2, axis=1, kind="stable")[:, :k]
    votes = y[nearest]
    n_labels = int(y.max()) + 1
    # argmax over the counts returns the smallest label among those with the most votes.
    return np.array([np.bincount(row, minlength=n_labels).argmax() for row in votes], dtype=np.int64)
```

#### What it costs
`O(M·N·D)` time for the distances and `O(M·N)` memory for the matrix; the sort adds `O(M·N log N)`. For `k ≪ N`,
`np.argpartition` brings selection to `O(M·N)`, but it does not preserve ties, so you would then sort only the
candidates. Training is free; prediction is the expensive part, which is why real systems use approximate nearest
neighbour indexes (HNSW, IVF) and process test points in chunks to bound memory.

#### Pitfalls
- The expansion can come out slightly negative from rounding (`-1e-12`). It does not matter for ranking, but clip at 0
  before a `sqrt`.
- `np.argsort` without `kind="stable"`: the default quicksort may reorder equal distances.
- Breaking a vote tie by the nearest neighbour's label (a common convention, but not this one).
- A Python double loop over test and training points: fine for 10 points, a timeout for thousands.
