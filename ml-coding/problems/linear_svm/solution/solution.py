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
