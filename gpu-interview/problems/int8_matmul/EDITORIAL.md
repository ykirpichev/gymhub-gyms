#### The idea
Do the inner products entirely in integers, then apply both scales once per output: since
`x[i, k] ≈ xq[i, k]·xs[i]` and `w[j, k] ≈ wq[j, k]·ws[j]`, the scales factor out of the sum over `k`.

#### A solution
```python
import numpy as np


def solve(xq, xs, wq, ws):
    x = np.asarray(xq, dtype=np.int64)
    w = np.asarray(wq, dtype=np.int64)
    acc = x @ w.T                                    # exact integer accumulation
    y = acc * np.asarray(xs, dtype=np.float64)[:, None] * np.asarray(ws, dtype=np.float64)[None, :]
    return (acc.tolist(), y.tolist())
```

#### On a real GPU
Tensor Cores multiply int8 by int8 and accumulate in int32, at about twice the FP16 rate. The scales go in the
*epilogue*: after the integer tile is done, each output is multiplied by its row's and column's scale, converted to
BF16 or FP16, and written. Int32 accumulation cannot overflow for realistic sizes: each product is at most
`127² ≈ 16,000`, so 2³¹ allows over 130,000 of them.

#### Pitfalls
- Accumulating in int8 or int16: overflows almost at once.
- Applying the scales before the sum (dequantizing every element): correct, but it gives up the integer speed.
- Mixing up which scale goes where: `xs` indexes rows of the output, `ws` columns.
