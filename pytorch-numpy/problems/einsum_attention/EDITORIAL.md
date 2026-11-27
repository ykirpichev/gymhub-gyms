#### The idea
Name the axes and let einsum do the transposes. Scores contract over `d` and keep `b, h, i, j`; the output contracts
over `j` and writes straight back to the `(B, T, H, D)` layout.

#### A solution
```python
import numpy as np


def solve(q, k, v):
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    s = np.einsum("bihd,bjhd->bhij", q, k) / np.sqrt(q.shape[-1])
    s = s - s.max(-1, keepdims=True)                 # stable softmax over j
    p = np.exp(s)
    p = p / p.sum(-1, keepdims=True)
    return np.einsum("bhij,bjhd->bihd", p, v)
```

#### What it costs
Two contractions of `B·H·T²·D` multiply-adds each, and a `B × H × T × T` score array in memory. Einsum on NumPy
may or may not route through BLAS (`optimize=True` helps); in PyTorch, `torch.einsum` lowers to batched matmuls.
Writing the strings is the skill: they make the layout explicit, which is where attention bugs live.

#### Pitfalls
- Contracting over `h` by reusing a letter (`"bihd,bjhd->bij"`): sums across heads.
- Softmax over the wrong axis: it is over keys `j`, the last axis of `s`.
- A Python loop over heads: correct, and the task rules it out.
