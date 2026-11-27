import numpy as np


def solve(logits, targets):
    x = np.asarray(logits, dtype=np.float64)
    t = np.asarray(targets, dtype=np.int64)
    shifted = x - x.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    picked = logp[np.arange(len(t)), t]
    return (picked, float(-picked.mean()))
