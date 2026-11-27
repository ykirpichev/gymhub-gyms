#### The idea
Mask with `-inf`, then the usual `m + log Σ exp(x − m)`. A fully masked row is the trap: its max is `-inf`, and
`x − m` is `-inf − (−inf) = NaN`. Replace a non-finite max by 0 before subtracting; the sum is then `Σ exp(−inf) = 0`
and `log 0 = −inf`, which is the right answer.

#### A solution
```python
import numpy as np


def solve(x, mask):
    x = np.where(np.asarray(mask, dtype=bool), np.asarray(x, dtype=np.float64), -np.inf)
    m = x.max(axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):                  # log(0) for fully masked rows
        return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).squeeze(1)
```

#### Where it shows up
Attention with padding or causal masks, when a query has no valid key (a fully padded row). Flash-attention kernels
special-case it the same way, and some return 0 for such rows instead of NaN so that one padded sequence cannot
poison a batch's gradients.

#### Pitfalls
- Masking with a large negative number such as `-1e9`: works in float64, overflows in FP16, and turns a fully masked
  row into a uniform average instead of `-inf`.
- Subtracting the max *before* masking: the max of the unmasked values is wrong.
