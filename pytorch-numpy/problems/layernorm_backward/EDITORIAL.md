#### The idea
Recompute the forward pieces (`μ`, `1/σ`, `x̂`), then use the simplified row formula. With `g = dy ⊙ γ`:
`dx = (1/σ)(g − mean(g) − x̂ ⊙ mean(g ⊙ x̂))`. The two subtracted terms are the gradient flowing back through the mean
and through the variance. `dγ` and `dβ` sum over rows, because every row uses the same `γ` and `β`.

#### A solution
```python
import numpy as np


def solve(x, gamma, beta, dy, eps):
    x, gamma, beta, dy = (np.asarray(a, dtype=np.float64) for a in (x, gamma, beta, dy))
    mu = x.mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(((x - mu) ** 2).mean(axis=1, keepdims=True) + eps)
    x_hat = (x - mu) * inv_std
    y = gamma * x_hat + beta
    g = dy * gamma
    dx = inv_std * (g - g.mean(axis=1, keepdims=True) - x_hat * (g * x_hat).mean(axis=1, keepdims=True))
    return (y, dx, (dy * x_hat).sum(axis=0), dy.sum(axis=0))
```

#### Checking it
Compare with autograd on random inputs (`torch.nn.functional.layer_norm` with `requires_grad`), or with finite
differences. Two properties make quick sanity checks: each row of `dx` sums to zero, and `dx` is orthogonal to `x̂`
in each row, because layer norm's output does not change if you shift or scale a row.

#### Pitfalls
- Forgetting the mean and variance paths and returning `dy · γ / σ`: right only if `μ` and `σ` were constants.
- Summing `dγ` over columns instead of rows.
