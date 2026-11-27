#### The idea
Floating-point addition rounds away the low bits of the smaller number. Kahan summation keeps those lost bits in a
compensation `c` and feeds them back into the next addition: `t − total` is what actually got added, and
subtracting `y` from it leaves (minus) the part that was rounded off.

#### A solution
```python
import numpy as np


def solve(values):
    f = np.float32
    naive, total, c = f(0.0), f(0.0), f(0.0)
    for value in values:
        x = f(value)
        naive = f(naive + x)
        y = f(x - c)
        t = f(total + y)
        c = f(f(t - total) - y)
        total = t
    return (float(naive), float(total))
```

#### Why it matters for ML
Naive summation's error grows with the number of terms; Kahan's stays at a couple of rounding units. It is why
Tensor Cores and reductions accumulate in FP32 even when inputs are BF16, and why some low-precision optimizers keep
Kahan-compensated BF16 weights so that tiny updates are not rounded away. Pairwise (tree) summation, which GPU
reductions do naturally, also keeps the error small: `O(log n)` instead of `O(n)`.

#### Pitfalls
- Letting NumPy promote to float64 (mixing a Python float into the expression): the test then sees no rounding at
  all. Wrap every intermediate result.
- Compilers with fast-math reorder `(t − total) − y` to 0 and remove the compensation; C code needs care.
