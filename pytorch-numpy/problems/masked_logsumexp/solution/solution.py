import numpy as np


def solve(x, mask):
    x = np.where(np.asarray(mask, dtype=bool), np.asarray(x, dtype=np.float64), -np.inf)
    m = x.max(axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True))).squeeze(1)
