"""Grader for rag_retrieval. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 2000}


def _scores(queries, docs):
    q = np.asarray(queries, dtype=np.float64)
    d = np.asarray(docs, dtype=np.float64)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    dn = np.linalg.norm(d, axis=1, keepdims=True)
    return (q / np.where(qn == 0, 1.0, qn)) @ (d / np.where(dn == 0, 1.0, dn)).T


def reference(queries, docs, k):
    q = np.asarray(queries, dtype=np.float64)
    d = np.asarray(docs, dtype=np.float64)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1] or k < 1:
        return "ValueError"
    scores = _scores(q, d)
    order = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, d.shape[0])]
    return order.astype(np.int64), np.take_along_axis(scores, order, axis=1)


def call(module, args):
    """Bad shapes or k must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError: k < 1 or the embedding widths differ" if want == "ValueError"
                                 else "raised ValueError, but the inputs are valid")
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (indices, scores), got {type(got).__name__}")
    indices, scores = got
    default_compare(scores, want[1], 1e-9, 1e-9, "scores")
    g = np.asarray(indices)
    if g.shape == want[0].shape and g.dtype.kind == "f" and np.all(g == np.round(g)):
        g = g.astype(np.int64)
    if g.shape == want[0].shape and g.dtype.kind in "iu" and not np.array_equal(g, want[0]):
        g = _forgive_rounding(g, want[0], case)
    try:
        default_compare(g, want[0], 0, 0, "indices")
    except AssertionError as exc:
        if g.shape == want[0].shape:
            # Scores match, so the documents differ only among equal scores.
            raise AssertionError(f"{exc}\n  equal scores must be ordered by lower document index first") from None
        raise


def _forgive_rounding(got, want, case):
    """Accept a different index where the two documents' scores differ only by float rounding (not an
    exact tie), so a different BLAS summation order cannot flip the verdict. Exact ties stay strict."""
    table = _scores(case["queries"], case["docs"])
    n = table.shape[1]
    got = got.copy()
    for i, t in zip(*np.nonzero(got != want)):
        j, w = int(got[i, t]), int(want[i, t])
        if 0 <= j < n and table[i, j] != table[i, w] and abs(table[i, j] - table[i, w]) <= 1e-12:
            got[i, t] = w
    return got


def _random(seed, q, n, d, k, name, duplicates=0):
    rng = np.random.default_rng(seed)
    docs = rng.standard_normal((n, d))
    queries = rng.standard_normal((q, d))
    if duplicates:
        # Rescaled copies of earlier documents: exactly equal cosine scores at different indices.
        src = rng.integers(0, n // 2, duplicates)
        dst = rng.choice(np.arange(n // 2, n), duplicates, replace=False)
        docs[dst] = docs[src] * 4.0  # a power of two keeps the normalized rows bit-identical
        docs[rng.integers(0, n, 3)] = 0.0
    return {"_name": name, "queries": queries.tolist(), "docs": docs.tolist(), "k": k}


def functional_cases():
    return [
        {"_name": "example", "queries": [[1, 0], [0, 0]], "docs": [[0, 2], [3, 3], [5, 0], [1, 0]], "k": 3},
        {"_name": "k = 1", "queries": [[1, 1, 0]], "docs": [[0, 0, 1], [2, 2, 0], [1, 0, 0]], "k": 1},
        {"_name": "k > N returns all", "queries": [[0.0, 1.0], [-1.0, 0.0]], "docs": [[1.0, 1.0], [0.0, -3.0], [-2.0, 0.5]], "k": 10},
        {"_name": "negative scores rank last", "queries": [[1.0, 0.0]], "docs": [[-1.0, 0.0], [0.0, 1.0], [-1.0, -1.0], [1.0, 1.0]], "k": 4},
        {"_name": "zero document", "queries": [[1.0, -1.0]], "docs": [[0.0, 0.0], [-1.0, 1.0], [0.0, 0.0], [0.0, 5.0]], "k": 3},
        {"_name": "ties at the cut-off", "queries": [[0.0, 1.0]],
         "docs": [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], "k": 3},
        _random(1, 3, 20, 5, 4, "random 3 queries, 20 docs"),
        _random(2, 50, 400, 16, 10, "random with duplicate and zero docs", duplicates=40),
        _random(3, 200, 5000, 32, 20, "random 200 queries, 5000 docs"),
        {"_name": "k = 0", "queries": [[1.0, 0.0]], "docs": [[1.0, 0.0]], "k": 0},
        {"_name": "width mismatch", "queries": [[1.0, 0.0, 0.0]], "docs": [[1.0, 0.0]], "k": 1},
    ]


def performance_case(seed: int = 0):
    return _random(seed, 200, 20000, 64, 10, "perf")


def build() -> Problem:
    return Problem(
        slug="rag_retrieval",
        title="Exact cosine top-k retrieval",
        difficulty="medium",
        summary="Rank documents by cosine similarity for a batch of queries and return the top k, with ties and zero vectors defined.",
        params=["queries", "docs", "k"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        call=call,
        compare=compare,
        tags=["numpy", "embeddings", "retrieval", "ranking"],
    )
