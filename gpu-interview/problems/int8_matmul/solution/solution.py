import numpy as np


def solve(xq, xs, wq, ws):
    x = np.asarray(xq, dtype=np.int64)
    w = np.asarray(wq, dtype=np.int64)
    acc = x @ w.T
    y = acc * np.asarray(xs, dtype=np.float64)[:, None] * np.asarray(ws, dtype=np.float64)[None, :]
    return (acc.tolist(), y.tolist())
