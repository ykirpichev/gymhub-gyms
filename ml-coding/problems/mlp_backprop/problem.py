"""Grader for mlp_backprop. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

NAMES = ["loss", "dW1", "db1", "dW2", "db2"]


def reference(X, y, W1, b1, W2, b2):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    W1, b1, W2, b2 = (np.asarray(a, dtype=np.float64) for a in (W1, b1, W2, b2))
    n = X.shape[0]
    z1 = X @ W1 + b1
    h = np.where(z1 > 0, z1, 0.0)
    logits = h @ W2 + b2
    m = logits.max(axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    log_probs = logits - lse
    loss = -log_probs[np.arange(n), y].sum() / n
    dlogits = np.exp(log_probs)
    dlogits[np.arange(n), y] -= 1.0
    dlogits = dlogits / n
    dz1 = (dlogits @ W2.T) * (z1 > 0)
    return float(loss), X.T @ dz1, dz1.sum(axis=0), h.T @ dlogits, dlogits.sum(axis=0)


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, tuple) or len(got) != 5:
        raise AssertionError(f"return a tuple (loss, dW1, db1, dW2, db2), got {type(got).__name__}")
    default_compare(got[0], want[0], 1e-8, 1e-7, "loss")
    for name, g, w in zip(NAMES[1:], got[1:], want[1:]):
        g_arr = np.asarray(g, dtype=np.float64) if not isinstance(g, str) else g
        if not isinstance(g_arr, str) and g_arr.shape == w.shape and len(case["y"]) > 1:
            scale = np.asarray(case["y"]).size
            if np.allclose(g_arr, w * scale, atol=1e-8, rtol=1e-6) and not np.allclose(w, 0, atol=1e-12):
                raise AssertionError(f"{name}: every value is N = {scale} times too large; "
                                     "the loss is a mean, so its gradient carries a 1/N")
        default_compare(g, w, 1e-8, 1e-6, name)


def _random(seed, n, d, hidden, c, name, scale=1.0):
    rng = np.random.default_rng(seed)
    return {
        "_name": name,
        "X": rng.standard_normal((n, d)).tolist(),
        "y": rng.integers(0, c, n).tolist(),
        "W1": (rng.standard_normal((d, hidden)) * scale / np.sqrt(d)).tolist(),
        "b1": (rng.standard_normal(hidden) * 0.1).tolist(),
        "W2": (rng.standard_normal((hidden, c)) * scale / np.sqrt(hidden)).tolist(),
        "b2": (rng.standard_normal(c) * 0.1).tolist(),
    }


def functional_cases():
    return [
        {"_name": "example", "X": [[1.0, 2.0], [-1.0, 0.5]], "y": [0, 1],
         "W1": [[0.5, -1.0], [0.25, 1.0]], "b1": [0.0, 0.5],
         "W2": [[1.0, -1.0], [0.5, -0.25]], "b2": [0.0, 0.0]},
        {"_name": "batch of one", "X": [[0.5, -1.0, 2.0]], "y": [2],
         "W1": [[1.0, 0.0], [0.0, -1.0], [0.5, 0.5]], "b1": [0.1, -0.2],
         "W2": [[0.3, -0.2, 0.1], [0.0, 0.4, -0.5]], "b2": [0.0, 0.1, -0.1]},
        # Hidden unit 0 has pre-activation exactly 0 for both samples: ReLU'(0) = 0 blocks its gradient.
        {"_name": "pre-activation exactly 0", "X": [[1.0, -1.0], [2.0, -2.0]], "y": [1, 0],
         "W1": [[1.0, 1.0], [1.0, 0.0]], "b1": [0.0, 0.0],
         "W2": [[2.0, -1.0], [1.0, -0.5]], "b2": [0.0, 0.0]},
        {"_name": "dead hidden unit", "X": [[1.0, 1.0], [2.0, 0.5], [0.5, 3.0]], "y": [0, 1, 1],
         "W1": [[-1.0, 0.5], [-1.0, 0.5]], "b1": [-0.5, 0.0],
         "W2": [[1.0, 2.0], [-1.0, 1.0]], "b2": [0.2, -0.2]},
        {"_name": "huge logits stay finite", "X": [[10.0, -5.0], [3.0, 8.0]], "y": [1, 0],
         "W1": [[20.0, -10.0], [5.0, 30.0]], "b1": [0.0, 1.0],
         "W2": [[30.0, -30.0], [-20.0, 25.0]], "b2": [0.0, 0.0]},
        {"_name": "one output class", "X": [[1.0, 2.0], [3.0, -1.0]], "y": [0, 0],
         "W1": [[1.0, 0.0], [0.0, 1.0]], "b1": [0.0, 0.0], "W2": [[1.0], [2.0]], "b2": [0.5]},
        _random(1, 16, 5, 8, 4, "random N=16 D=5 H=8 C=4"),
        _random(2, 64, 10, 32, 6, "random N=64 D=10 H=32 C=6"),
        _random(3, 1, 7, 12, 3, "random batch of one"),
    ]


def build() -> Problem:
    return Problem(
        slug="mlp_backprop",
        title="MLP forward and backward pass",
        difficulty="hard",
        summary="Backpropagation by hand through a one-hidden-layer ReLU network with a softmax cross-entropy output.",
        params=["X", "y", "W1", "b1", "W2", "b2"],
        reference=reference,
        functional_cases=functional_cases,
        compare=compare,
        tags=["numpy", "neural networks", "backpropagation"],
    )
