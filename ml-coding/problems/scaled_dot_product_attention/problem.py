"""Grader for scaled_dot_product_attention. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(q, k, v, mask):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
    m = scores.max(axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(scores - m)
    total = e.sum(axis=-1, keepdims=True)
    weights = e / np.where(total == 0, 1.0, total)
    return weights @ v, weights


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (output, weights), got {type(got).__name__}")
    for value, name in ((got[0], "output"), (got[1], "weights")):
        if np.isnan(np.asarray(value, dtype=np.float64)).any():
            raise AssertionError(f"{name} contains NaN (fully masked row, or exp overflow without subtracting the max?)")
    default_compare(got[1], want[1], 1e-9, 1e-7, "weights")
    default_compare(got[0], want[0], 1e-9, 1e-7, "output")


def _causal(n):
    return np.tril(np.ones((n, n), dtype=bool)).tolist()


def _random(seed, batch, lq, lk, dk, dv, name, mask_kind=None, scale=1.0):
    rng = np.random.default_rng(seed)
    case = {
        "_name": name,
        "q": (rng.standard_normal(batch + (lq, dk)) * scale).tolist(),
        "k": (rng.standard_normal(batch + (lk, dk)) * scale).tolist(),
        "v": rng.standard_normal(batch + (lk, dv)).tolist(),
        "mask": None,
    }
    if mask_kind == "causal":
        case["mask"] = _causal(lq)
    elif mask_kind == "random":
        case["mask"] = (rng.random(batch + (lq, lk)) < 0.7).tolist()
    return case


def functional_cases():
    return [
        {"_name": "example",
         "q": [[1.0, 0.0], [0.0, 1.0]], "k": [[1.0, 0.0], [0.0, 1.0]], "v": [[1.0, 2.0], [3.0, 4.0]],
         "mask": [[True, False], [True, True]]},
        {"_name": "no mask", "q": [[1.0, 2.0, 0.5]], "k": [[0.0, 1.0, 1.0], [2.0, -1.0, 0.0]],
         "v": [[1.0], [-1.0]], "mask": None},
        {"_name": "scale uses d_k, not d_v", "q": [[2.0, 0.0, 0.0, 0.0]], "k": [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
         "v": [[1.0, 0.0], [0.0, 1.0]], "mask": None},
        {"_name": "fully masked row gives zeros",
         "q": [[1.0, 1.0], [0.5, -0.5]], "k": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
         "v": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
         "mask": [[False, False, False], [False, True, True]]},
        {"_name": "huge scores need a stable softmax",
         "q": [[1000.0, 0.0], [0.0, -1000.0]], "k": [[1000.0, 0.0], [-1000.0, 0.0], [0.0, 1000.0]],
         "v": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "mask": [[True, True, True], [True, False, True]]},
        _random(1, (), 4, 4, 3, 3, "causal 4x4", "causal"),
        _random(2, (2, 3), 5, 7, 4, 6, "batch (2, 3), Lq != Lk, random mask", "random"),
        _random(3, (3,), 6, 6, 8, 5, "batch 3, causal mask broadcast over batch", "causal"),
        _random(4, (2,), 3, 5, 4, 4, "batch 2, no mask, large values", None, scale=30.0),
        _random(5, (4, 4), 64, 64, 32, 32, "batch (4, 4), 64 x 64, causal", "causal"),
    ]


def build() -> Problem:
    return Problem(
        slug="scaled_dot_product_attention",
        title="Scaled dot-product attention",
        difficulty="medium",
        summary="Batched softmax(QK^T / sqrt(d_k)) V with a boolean mask applied before a stable softmax.",
        params=["q", "k", "v", "mask"],
        reference=reference,
        functional_cases=functional_cases,
        compare=compare,
        tags=["numpy", "attention", "transformer", "numerical stability"],
    )
