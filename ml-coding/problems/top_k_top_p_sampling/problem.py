"""Grader for top_k_top_p_sampling. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _sample(logits, temperature, k, p, u):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty 1-D sequence")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be >= 0")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be >= 0")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive int")
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")
    if isinstance(u, bool) or not isinstance(u, (int, float)) or not np.isfinite(u) or u < 0 or u >= 1:
        raise ValueError("u must be in [0, 1)")

    vocab = int(logits.shape[0])
    if float(temperature) == 0.0:
        token = int(np.argmax(logits))
        probs = np.zeros(vocab, dtype=np.float64)
        probs[token] = 1.0
        return probs, token

    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    probs = exp / np.sum(exp)

    def rank(weights):
        return np.lexsort((np.arange(weights.shape[0]), -weights))

    chosen = rank(probs)[: min(k, vocab)]
    filtered = np.zeros(vocab, dtype=np.float64)
    filtered[chosen] = probs[chosen]
    filtered /= np.sum(filtered)

    order = rank(filtered)
    order = order[filtered[order] > 0.0]
    cumulative = np.cumsum(filtered[order])
    count = int(np.searchsorted(cumulative, float(p), side="left")) + 1
    count = min(max(count, 1), int(order.shape[0]))
    kept = order[:count]
    out = np.zeros(vocab, dtype=np.float64)
    out[kept] = filtered[kept]
    out /= np.sum(out)

    cdf = np.cumsum(out)
    token = int(np.searchsorted(cdf, float(u), side="right"))
    if token == vocab:
        token = int(np.flatnonzero(out)[-1])
    return out, token


def reference(logits, temperature, k, p, u):
    try:
        return _sample(logits, temperature, k, p, u)
    except ValueError:
        return "ValueError"


def call(module, args):
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError" if want == "ValueError" else "raised ValueError on a valid input"
            )
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (probs, token), got {type(got).__name__}")
    probs = np.asarray(got[0], dtype=np.float64)
    if probs.shape != want[0].shape:
        raise AssertionError(f"probs: shape {list(probs.shape)} != expected {list(want[0].shape)}")
    if not np.isfinite(probs).all():
        raise AssertionError("probs contain non-finite values (divide by temperature, then subtract the max before exp)")
    total = float(probs.sum())
    if abs(total - 1.0) > 1e-6:
        raise AssertionError(f"probs sum to {total:.6g}, expected 1 after renormalizing the tokens top-k and top-p kept")
    kept_got = np.flatnonzero(probs > 1e-8).tolist()
    kept_want = np.flatnonzero(np.asarray(want[0]) > 1e-8).tolist()
    if kept_got != kept_want:
        raise AssertionError(
            f"kept token indices {kept_got}, expected {kept_want} "
            "(temperature, then top-k, renormalize, then top-p; equal probabilities keep the lower index)"
        )
    default_compare(got[0], want[0], 1e-9, 1e-7, "probs")
    default_compare(got[1], want[1], 1e-9, 1e-7, "token")


def _from_probs(probs, temperature, k, p, u, name):
    logits = np.log(np.asarray(probs, dtype=np.float64)).tolist()
    return {"_name": name, "logits": logits, "temperature": temperature, "k": k, "p": p, "u": u}


def _random(seed, vocab, name, scale, temperature, k=None):
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal(vocab) * scale).tolist()
    if k is None:
        k = int(rng.integers(1, vocab + 1))
    return {
        "_name": name,
        "logits": logits,
        "temperature": float(temperature),
        "k": int(k),
        "p": float(rng.uniform(0.15, 1.0)),
        "u": float(rng.random()),
    }


def functional_cases():
    return [
        {"_name": "example", "logits": [1.0, 2.0, 3.0], "temperature": 2.0, "k": 2, "p": 0.8, "u": 0.0},
        {"_name": "temperature 0, lowest-index tie", "logits": [1.0, 3.0, 3.0, 2.0],
         "temperature": 0.0, "k": 2, "p": 0.4, "u": 0.9},
        {"_name": "top-k tie keeps the lower index", "logits": [0.0, 5.0, 5.0, 5.0],
         "temperature": 1.0, "k": 2, "p": 1.0, "u": 0.0},
        {"_name": "top-p tie keeps the lower indices", "logits": [0.0, 0.0, 0.0, 0.0],
         "temperature": 1.0, "k": 4, "p": 0.5, "u": 0.0},
        _from_probs([0.50, 0.30, 0.15, 0.05], 1.0, 2, 0.6, 0.0, "top-k, renormalize, then top-p"),
        _from_probs([0.40, 0.35, 0.25], 1.0, 3, 0.5, 0.6, "top-p keeps the token that crosses p"),
        {"_name": "inverse CDF boundary", "logits": [0.0, 0.0], "temperature": 1.0, "k": 2, "p": 1.0, "u": 0.5},
        {"_name": "huge logits stay finite", "logits": [1000.0, 0.0, -1000.0],
         "temperature": 1.0, "k": 3, "p": 1.0, "u": 0.0},
        _random(1, 20, "random vocab 20", 2.0, 0.5),
        _random(2, 36, "random vocab 36, k above vocab", 2.5, 1.7, k=41),
        {"_name": "k <= 0", "logits": [0.0, 1.0], "temperature": 0.0, "k": -1, "p": 1.0, "u": 0.0},
        {"_name": "p == 0", "logits": [0.0, 1.0], "temperature": 1.0, "k": 1, "p": 0.0, "u": 0.0},
        {"_name": "p > 1", "logits": [0.0, 1.0], "temperature": 0.0, "k": 1, "p": 1.2, "u": 0.0},
        {"_name": "u == 1", "logits": [0.0, 1.0], "temperature": 0.0, "k": 1, "p": 1.0, "u": 1.0},
        {"_name": "negative temperature", "logits": [0.0, 1.0], "temperature": -1.0, "k": 1, "p": 1.0, "u": 0.0},
    ]


def build() -> Problem:
    return Problem(
        slug="top_k_top_p_sampling",
        title="Temperature, top-k, and top-p sampling",
        difficulty="medium",
        summary="Temperature-scaled softmax, then top-k, then top-p, renormalized, plus one inverse-CDF sample.",
        params=["logits", "temperature", "k", "p", "u"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "decoding", "sampling"],
    )
