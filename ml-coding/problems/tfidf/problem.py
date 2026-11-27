"""Grader for tfidf. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import math
from collections import Counter

import numpy as np

from gymkit.spec import Problem


def reference(docs):
    vocab = sorted({token for doc in docs for token in doc})
    index = {token: j for j, token in enumerate(vocab)}
    n = len(docs)
    counts = np.zeros((n, len(vocab)), dtype=np.float64)
    for i, doc in enumerate(docs):
        for token, c in Counter(doc).items():
            counts[i, index[token]] = c
    df = (counts > 0).sum(axis=0)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    weights = counts * idf
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    return vocab, weights / np.where(norms == 0, 1.0, norms)


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (vocab, matrix), got {type(got).__name__}")
    vocab, matrix = got
    if list(vocab) != want[0]:
        raise AssertionError(f"vocab: got {list(vocab)!r:.200}, expected {want[0]!r:.200} (sorted unique tokens)")
    default_compare(matrix, want[1], 1e-9, 1e-7, "matrix")
    if np.asarray(matrix).dtype.kind != "f":
        raise AssertionError(f"matrix: expected a float array, got dtype {np.asarray(matrix).dtype}")


def _random(seed, n_docs, max_len, vocab_size, name):
    rng = np.random.default_rng(seed)
    words = [f"w{k:03d}" for k in range(vocab_size)]
    # Zipf-like frequencies so df varies a lot across tokens.
    p = 1.0 / np.arange(1, vocab_size + 1)
    p /= p.sum()
    docs = [[words[k] for k in rng.choice(vocab_size, size=int(rng.integers(0, max_len + 1)), p=p)]
            for _ in range(n_docs)]
    return {"_name": name, "docs": docs}


def functional_cases():
    return [
        {"_name": "example", "docs": [["the", "cat", "sat"], ["the", "dog"], ["cat", "cat"]]},
        {"_name": "single document", "docs": [["a", "b", "a"]]},
        {"_name": "repeated token counts once for df", "docs": [["x", "x", "x", "y"], ["y"], ["z"]]},
        {"_name": "empty document gives a zero row", "docs": [["a"], [], ["b", "a"]]},
        {"_name": "every document empty", "docs": [[], []]},
        {"_name": "no documents", "docs": []},
        {"_name": "vocabulary is sorted, not first-seen", "docs": [["zeta", "alpha", "Beta"], ["mid", "zeta"]]},
        {"_name": "token in every document", "docs": [["a", "b"], ["a"], ["a", "c", "c"]]},
        _random(1, 6, 8, 10, "random 6 docs"),
        _random(2, 300, 40, 200, "random 300 docs"),
    ]


def build() -> Problem:
    return Problem(
        slug="tfidf",
        title="TF-IDF matrix",
        difficulty="medium",
        summary="Build a sorted vocabulary and an L2-normalized TF-IDF matrix with smoothed IDF from tokenized documents.",
        params=["docs"],
        reference=reference,
        functional_cases=functional_cases,
        compare=compare,
        tags=["numpy", "nlp", "feature extraction"],
    )
