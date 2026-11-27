import numpy as np


def solve(logits, labels):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n, c = x.shape
    if y.shape != (n,) or np.any(y < 0) or np.any(y >= c):
        raise ValueError("labels must be N class indices in [0, C)")
    m = x.max(axis=1, keepdims=True)
    shifted = x - m
    exp = np.exp(shifted)
    total = exp.sum(axis=1, keepdims=True)
    probs = exp / total
    log_probs = shifted - np.log(total)
    loss = float(-log_probs[np.arange(n), y].mean())
    return probs, loss
