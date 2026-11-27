#### The idea
The hinge loss `max(0, 1 − y·f(x))` is zero for points safely on the right side (margin ≥ 1) and linear otherwise.
Its subgradient is `−y·x` for violators and 0 for the rest, so each step only looks at the violators, plus the
regulariser's `λw`. Compute the margins from the **current** parameters, pick the violators with a boolean mask, and
update `w` and `b` from the same snapshot.

#### A solution
```python
import numpy as np


def solve(X, y, lam, lr, steps):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all((y == 1) | (y == -1)):
        raise ValueError("labels must be -1 or +1")
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        margins = y * (X @ w + b)
        active = margins < 1  # margin exactly 1 contributes no hinge gradient
        grad_w = lam * w - (y[active] @ X[active]) / n
        grad_b = -y[active].sum() / n
        w = w - lr * grad_w
        b = b - lr * grad_b
    hinge = np.maximum(0.0, 1 - y * (X @ w + b)).mean()
    return w, b, float(hinge + 0.5 * lam * w @ w)
```

`y[active] @ X[active]` is `Σ yᵢ xᵢ` over violators in one call.

#### What it costs
`O(N·D)` per step. The objective is convex but not differentiable at margin exactly 1, so plain subgradient descent
with a fixed step only gets near the optimum. Pegasos uses a decaying step `1/(λt)`; dual solvers (SMO, liblinear)
converge faster and expose the support vectors, the points with margin ≤ 1 that define the boundary.

#### Pitfalls
- `margins <= 1` instead of `< 1`: a point exactly on the margin then pulls the boundary.
- Regularising the bias.
- Forgetting `1/N` on the hinge part but keeping it on the loss, so the gradient and the objective disagree.
- Multiplying `y` by `X` row-wise and summing with a Python loop.
