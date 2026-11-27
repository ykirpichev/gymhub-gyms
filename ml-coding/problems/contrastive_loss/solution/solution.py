import numpy as np


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _cross_entropy_diagonal(logits):
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return float(np.mean(lse - np.diag(logits)))


def solve(img, txt, temperature):
    img = np.asarray(img, dtype=np.float64)
    txt = np.asarray(txt, dtype=np.float64)
    if img.ndim != 2 or img.shape != txt.shape or not temperature > 0:
        raise ValueError("img and txt must both be N x d, and temperature must be positive")
    logits = _normalize(img) @ _normalize(txt).T / temperature
    return (_cross_entropy_diagonal(logits) + _cross_entropy_diagonal(logits.T)) / 2
