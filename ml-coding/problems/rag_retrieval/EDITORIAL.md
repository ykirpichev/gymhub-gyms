#### The idea
Normalise every query and document once (a zero vector keeps norm 1 so it stays zero), and cosine similarity becomes
a plain dot product. All `Q × N` scores are one matmul. A stable argsort of the negated scores gives "highest first,
lower index on ties", and `np.take_along_axis` pulls the matching scores.

#### A solution
```python
import numpy as np


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def solve(queries, docs, k):
    q = np.asarray(queries, dtype=np.float64)
    d = np.asarray(docs, dtype=np.float64)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1] or k < 1:
        raise ValueError("queries and docs must be 2-D with the same width, and k >= 1")
    scores = _normalize(q) @ _normalize(d).T
    order = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, d.shape[0])]
    return order, np.take_along_axis(scores, order, axis=1)
```

#### What it costs
`O(Q·N·d)` for the scores and `O(Q·N log N)` for the sort. With `np.argpartition` the selection is `O(Q·N)`, then sort
only the `k` survivors (ties need care at the cut). Exact search is the baseline you measure recall against. At
millions of documents you move to an approximate index: IVF (cluster, then scan a few clusters), HNSW (graph walk), or
product quantisation to shrink vectors 10 to 30 times. Those trade a little recall for orders of magnitude in latency.

#### Pitfalls
- Dividing by a zero norm, which turns a zero query into a row of `nan`.
- Default `np.argsort` (quicksort) for the order, which does not keep lower indices first on ties.
- Normalising inside a Python loop over documents.
- Returning `k` columns when `k > N`.
