import numpy as np


def solve(X, y, W1, b1, W2, b2):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    W1, b1, W2, b2 = (np.asarray(a, dtype=np.float64) for a in (W1, b1, W2, b2))
    n = X.shape[0]

    # Forward, keeping what the backward pass needs.
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    logits = h @ W2 + b2
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = float(-log_probs[np.arange(n), y].mean())

    # Backward.
    dlogits = np.exp(log_probs)
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ W2.T
    dz1 = dh * (z1 > 0)  # ReLU'(0) = 0
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return loss, dW1, db1, dW2, db2
