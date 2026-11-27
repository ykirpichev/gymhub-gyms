"""Grader for softmax_cross_entropy. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(logits, labels):
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n, c = x.shape
    if np.any(y < 0) or np.any(y >= c):
        return "ValueError"
    m = x.max(axis=1, keepdims=True)
    exp = np.exp(x - m)
    total = exp.sum(axis=1, keepdims=True)
    log_probs = x - m - np.log(total)
    return exp / total, float(-log_probs[np.arange(n), y].mean())


def call(module, args):
    """Invalid labels must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError("expected ValueError for an out-of-range label" if want == "ValueError"
                                 else "raised ValueError, but every label is valid")
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (probs, loss), got {type(got).__name__}")
    default_compare(got[0], want[0], 1e-9, 1e-7, "probs")
    default_compare(got[1], want[1], 1e-9, 1e-7, "loss")


def _random(seed, n, c, scale, name):
    rng = np.random.default_rng(seed)
    return {"_name": name, "logits": (rng.standard_normal((n, c)) * scale).tolist(), "labels": rng.integers(0, c, n).tolist()}


def functional_cases():
    return [
        {"_name": "example", "logits": [[1.0, 2.0, 3.0]], "labels": [2]},
        {"_name": "one class", "logits": [[5.0], [-3.0]], "labels": [0, 0]},
        {"_name": "huge logits", "logits": [[1e4, 0.0, -1e4], [1000.0, 1001.0, 999.0]], "labels": [2, 1]},
        {"_name": "very negative", "logits": [[-1e4, -1e4 + 1.0]], "labels": [0]},
        {"_name": "equal logits", "logits": [[0.0] * 4, [7.0] * 4], "labels": [3, 0]},
        _random(1, 8, 5, 1.0, "random 8x5"),
        _random(2, 200, 50, 30.0, "random 200x50, wide range"),
        {"_name": "label too large", "logits": [[1.0, 2.0]], "labels": [2]},
        {"_name": "negative label", "logits": [[1.0, 2.0], [0.0, 0.0]], "labels": [0, -1]},
    ]


def build() -> Problem:
    return Problem(
        slug="softmax_cross_entropy",
        title="Stable softmax and cross-entropy",
        difficulty="easy",
        summary="Row-wise softmax and mean cross-entropy that survive huge logits, with label validation.",
        params=["logits", "labels"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "numerical stability", "probability"],
    )
