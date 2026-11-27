#### The idea
Online softmax, with a vector accumulator. For each query row, walk the key blocks keeping `m` (running max),
`l` (running sum of `exp(s − m)`) and `acc` (running sum of `exp(s − m)·v`). When a block raises the max, rescale
*both* `l` and `acc` by `alpha = exp(m_old − m_new)`. Divide at the end.

#### A solution
```python
import math
import numpy as np


def solve(Q, K, V, block, causal):
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (Q, K, V))
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2 or q.shape[1] != k.shape[1] or k.shape[0] != v.shape[0]:
        raise ValueError("Q is n x d, K is m x d, V is m x dv")
    if block < 1 or (causal and q.shape[0] != k.shape[0]):
        raise ValueError("bad block or causal shapes")
    n, d = q.shape
    keys, dv = v.shape
    scale = 1.0 / math.sqrt(d)
    out, lse = np.zeros((n, dv)), np.zeros(n)
    for i in range(n):
        m, l, acc = -math.inf, 0.0, np.zeros(dv)
        for start in range(0, keys, block):
            stop = min(start + block, keys)
            if causal:
                stop = min(stop, i + 1)              # keys j <= i only
            if stop <= start:
                continue                             # the whole block is in the future
            s = (k[start:stop] @ q[i]) * scale
            m_new = max(m, float(s.max()))
            alpha = math.exp(m - m_new)
            p = np.exp(s - m_new)
            l = alpha * l + float(p.sum())
            acc = alpha * acc + p @ v[start:stop]
            m = m_new
        out[i] = acc / l
        lse[i] = m + math.log(l)
    return (out, lse)
```

#### Why the GPU version is fast
The real kernel does this for a *tile* of query rows at once, so `s` is a small matrix computed by a Tensor Core
matmul, and `K`, `V` tiles are loaded into shared memory once per query tile. HBM traffic is `Q, K, V, O` plus the
`lse` vector, linear in sequence length; the `n × m` score matrix exists only in registers and shared memory. The
backward pass recomputes the scores from `Q`, `K` and `lse` instead of storing them, which is the other half of the
memory saving.

#### Pitfalls
- Forgetting to rescale `acc` when the max changes: the classic bug, and it passes when one block holds all keys.
- Causal blocks at the boundary: a block can be partly visible; clip it to `i + 1` rather than skipping it.
- Returning `log(l)` instead of `m + log(l)` for the log-sum-exp.
