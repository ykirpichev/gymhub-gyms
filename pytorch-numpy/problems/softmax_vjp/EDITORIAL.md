#### The idea
The softmax Jacobian is `diag(s) − s sᵀ`. Multiplying it by `g` gives `s ⊙ g − s (s·g)`, which factors to
`s ⊙ (g − s·g)`: one dot product per row, no `C × C` matrix.

#### A solution
```python
import numpy as np


def solve(x, g):
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    s = e / e.sum(axis=1, keepdims=True)
    dx = s * (g - (g * s).sum(axis=1, keepdims=True))
    return (s, dx)
```

#### Why it matters
Autograd never builds Jacobians: every backward function is a vector-Jacobian product like this one. Building the
`C × C` Jacobian per row costs `O(C²)` memory and time; with a 128,000-word vocabulary that is 16 billion entries per
row. The VJP is `O(C)`. A quick sanity check: each row of `dx` sums to zero, because adding a constant to `x` does
not change the softmax.

#### Pitfalls
- Using `g · s` summed over the whole batch instead of per row.
- Recomputing the softmax without the max subtraction in the backward pass.
