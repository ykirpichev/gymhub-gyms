#### The idea
Symmetric per-row (per-channel) quantization: map each row's largest magnitude to 127, so the scale is
`max|w| / 127`, divide, round to nearest even, clip. An all-zero row gets scale 1 so the division is safe.

#### A solution
```python
import numpy as np


def solve(W):
    w = np.asarray(W, dtype=np.float64)
    peak = np.abs(w).max(axis=1)
    scales = np.where(peak > 0, peak / 127.0, 1.0)
    q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int64)
    return (q.tolist(), scales.tolist())
```

#### Why per row
One scale for the whole tensor lets a single outlier weight set the step size for everyone, and small rows round to
almost nothing. A scale per output row (per channel) follows each row's range, and it costs one float per row,
which is applied once after the integer dot product. Group-wise scales (one per 64 or 128 values) go further and are
standard for 4-bit weights.

The error per element is at most half a step, `scale / 2`. Activations are harder than weights: a few channels carry
much larger values (outliers), which is why methods like SmoothQuant move that range into the weights first.

#### Pitfalls
- Using −128: symmetric schemes stop at −127 so that `q` and `−q` are both representable.
- Python's `round` and NumPy's `np.round` round half to even; `int(x + 0.5)` does not, and fails the `63.5` case.
- Dividing by a zero scale for an all-zero row.
