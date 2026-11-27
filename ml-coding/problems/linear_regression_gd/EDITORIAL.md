#### The idea
Write the residual `e = Xw + b − y` once per step. The MSE is `mean(e²)`, and its gradients follow from the chain rule:

$$\frac{\partial L}{\partial w} = \frac{2}{N} X^\top e, \qquad \frac{\partial L}{\partial b} = \frac{2}{N} \sum_i e_i$$

Both use the same `e`, computed from the parameters **before** the step, so `w` and `b` move together. Record the loss
before each update, then once more at the end, which gives the `steps + 1` values.

#### A solution
```python
import numpy as np


def solve(X, y, lr, steps):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    if y.shape != (n,):
        raise ValueError("y must have one target per row of X")
    w = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(steps):
        err = X @ w + b - y
        losses.append(float(np.mean(err ** 2)))
        w = w - lr * (2.0 / n) * (X.T @ err)
        b = b - lr * (2.0 / n) * err.sum()
    err = X @ w + b - y
    losses.append(float(np.mean(err ** 2)))
    return w, float(b), losses
```

#### What it costs
Each step is two matrix-vector products, `O(N·D)`. The whole fit is `O(steps·N·D)`. The closed form
`(XᵀX)⁻¹Xᵀy` costs `O(N·D² + D³)` once, which wins for small `D` and loses when `D` is large or the data does not fit in
memory (then you use mini-batch SGD instead).

Gradient descent converges only if `lr < 2 / λ_max`, where `λ_max` is the largest eigenvalue of `(2/N)XᵀX`. Features
in the thousands make `λ_max` huge, so the same learning rate that works on scaled data diverges. That is why you
standardise features in practice.

#### Pitfalls
- Dropping the factor 2 or the `1/N`: the fit still converges, to the same place, at a different speed, and every
  number in the test is off.
- Updating `w` and then computing `b`'s gradient from the new `w`.
- Looping over rows in Python: correct, and hundreds of times slower than `X.T @ err`.
- Returning `steps` losses instead of `steps + 1`.
