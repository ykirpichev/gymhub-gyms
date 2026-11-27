#### The idea
Expand the square: `‖x − y‖² = ‖x‖² − 2 x·y + ‖y‖²`. The middle term for all pairs is one matrix product `X Yᵀ`; the
two norms are vectors that broadcast as a column and a row.

#### A solution
```python
import numpy as np


def solve(X, Y):
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    d = (x * x).sum(1)[:, None] - 2 * x @ y.T + (y * y).sum(1)[None, :]
    return np.maximum(d, 0.0)
```

#### What it costs
`O(NMD)` FLOPs either way, but the broadcast version `((x[:, None] - y[None]) ** 2).sum(-1)` builds an `N × M × D`
temporary: 8 GB for 1,000 × 1,000 × 1,000 in float64. The expansion needs `N × M` memory and runs on an optimized
matmul (BLAS on CPU, Tensor Cores on GPU). `torch.cdist` uses the same trick for large inputs.

#### Pitfalls
- Cancellation: for nearby points the three terms are large and nearly cancel, so the result can be slightly
  negative. Clamp at 0, and use float64 or the direct formula when exact small distances matter.
- `np.sqrt` of an unclamped result gives NaN.
