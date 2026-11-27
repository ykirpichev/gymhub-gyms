#### The idea
Two numerically dangerous pieces, two fixes:

1. **Sigmoid.** `1 / (1 + exp(−z))` overflows for large negative `z`. Compute `e = exp(−|z|)`, which is always in
   `(0, 1]`, and pick the form that suits the sign: `1/(1+e)` for `z ≥ 0`, `e/(1+e)` for `z < 0`.
2. **Loss.** Expand the cross-entropy in terms of `z`: `−[y log σ(z) + (1−y) log(1−σ(z))] = log(1 + e^z) − y·z`, and
   `log(1 + e^z)` is `np.logaddexp(0, z)`, which is stable for any `z`.

The gradient is the famous clean one: `∂J/∂w = Xᵀ(p − y)/N + λw`, `∂J/∂b = mean(p − y)`. No regularisation on `b`.

#### A solution
```python
import numpy as np


def sigmoid(z):
    e = np.exp(-np.abs(z))  # in (0, 1]: never overflows
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def solve(X, y, lr, steps, l2=0.0, threshold=0.5):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    if y.shape != (n,) or not np.all((y == 0) | (y == 1)):
        raise ValueError("y must hold N labels, each 0 or 1")
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        err = sigmoid(X @ w + b) - y
        w = w - lr * (X.T @ err / n + l2 * w)
        b = b - lr * err.mean()
    z = X @ w + b
    # -[y log p + (1 - y) log(1 - p)] = log(1 + e^z) - y z, with log(1 + e^z) = logaddexp(0, z)
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w))
    preds = (sigmoid(z) >= threshold).astype(np.int64)
    return w, float(b), loss, preds
```

#### What it costs
`O(N·D)` per step, like linear regression. The objective is convex, so gradient descent with a small enough learning
rate reaches the global optimum. Without `l2`, on separable data, `w` grows forever and the loss tends to 0: L2 is what
keeps the solution finite.

#### Pitfalls
- `np.log(p)` with `p` rounded to exactly 0 or 1: `-inf`, then `nan` in the mean.
- `np.where(z >= 0, 1/(1+np.exp(-z)), ...)`: `np.where` evaluates both branches, so the overflow still happens. That
  is why the solution computes `exp(-|z|)` first.
- Regularising the bias, or adding `λw` to the gradient but `λ‖w‖²` (without the ½) to the loss.
- Accepting ±1 labels silently. The loss formula assumes 0/1, so validate them.
