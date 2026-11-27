import numpy as np


def _normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def solve(queries, docs, k):
    q = np.asarray(queries, dtype=np.float64)
    d = np.asarray(docs, dtype=np.float64)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1] or k < 1:
        raise ValueError("queries and docs must be 2-D with the same width, and k >= 1")
    scores = _normalize(q) @ _normalize(d).T
    order = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, d.shape[0])]
    return order, np.take_along_axis(scores, order, axis=1)
