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
