#### The idea
Softmax does not change when you add the same constant to every logit, because the constant cancels between the
numerator and the denominator. Subtract the row maximum first: the largest shifted logit is 0, so `exp` stays in
`(0, 1]` and never overflows, and the denominator is at least 1, so it never underflows to 0.

For the loss, never take `log` of a probability. Work in log space: `log softmax(x)_y = (x_y − m) − log Σ exp(x − m)`.
That is the log-sum-exp trick, and it stays finite even when the right class's probability rounds to 0.

#### A solution
```python
import numpy as np


def solve(logits, labels):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n, c = x.shape
    if y.shape != (n,) or np.any(y < 0) or np.any(y >= c):
        raise ValueError("labels must be N class indices in [0, C)")
    m = x.max(axis=1, keepdims=True)
    shifted = x - m
    exp = np.exp(shifted)
    total = exp.sum(axis=1, keepdims=True)
    probs = exp / total
    log_probs = shifted - np.log(total)
    loss = float(-log_probs[np.arange(n), y].mean())
    return probs, loss
```

`log_probs` is computed once and reused, so the loss never sees a 0 inside a `log`.

#### What it costs
`O(N·C)` time and memory: one pass for the max, one for `exp`, one for the sum. On a GPU this is exactly the kernel
that a fused cross-entropy (or an online softmax) avoids materialising when `C` is a 100k-token vocabulary.

#### Pitfalls
- `np.log(probs[range(n), y])`: correct for small logits, `-inf` as soon as a probability rounds to 0.
- Subtracting the **global** max instead of the per-row max: one large row pushes every other row into underflow.
- Forgetting `keepdims=True`, so `x - m` broadcasts `(N, C) − (N,)` the wrong way or fails.
- Returning the summed loss instead of the mean, or a 0-d array instead of a Python `float`.
- Not checking label range: NumPy's negative indexing silently reads the last class for `-1`.
