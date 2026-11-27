"""Grader for pca. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

ATOL = 1e-6
RTOL = 1e-5


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _orient(components):
    oriented = np.array(components, dtype=np.float64, copy=True)
    for row in oriented:
        index = int(np.argmax(np.abs(row)))
        if row[index] < 0:
            row *= -1
    return oriented


def reference(X, k):
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        return "ValueError"
    n, d = data.shape
    if not _is_int(k) or k < 1 or k > d or k > n - 1:
        return "ValueError"
    centered = data - data.mean(axis=0)
    if np.all(centered == 0):
        return "ValueError"
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular ** 2
    return _orient(vt[:k]), variance[:k] / variance.sum()


def _fit(data, k, center):
    matrix = np.asarray(data, dtype=np.float64)
    if center:
        matrix = matrix - matrix.mean(axis=0)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    variance = singular ** 2
    total = float(variance.sum())
    ratio = variance[:k] / total if total else np.zeros(k)
    return _orient(vt[:k]), ratio


def call(module, args):
    """k out of range, or a matrix with no variation, must raise ValueError."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError: k must lie in 1 .. min(D, N-1), and the rows of X must not all be equal"
                if want == "ValueError" else "raised ValueError, but X and k are valid")
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return (components, explained_variance_ratio), got {type(got).__name__}")
    try:
        components = np.asarray(got[0], dtype=np.float64)
        ratio = np.asarray(got[1], dtype=np.float64)
    except (ValueError, TypeError):
        raise AssertionError("components and explained_variance_ratio must be numeric arrays") from None
    want_components, want_ratio = want
    if components.shape == want_components.shape and components.ndim == 2:
        if len(want_components) > 1 and np.allclose(components, want_components[::-1], atol=ATOL, rtol=RTOL):
            raise AssertionError("components are in ascending variance order. Return the largest variance first.")
        wrong_sign = [
            index for index in range(len(want_components))
            if np.allclose(components[index], -want_components[index], atol=ATOL, rtol=RTOL)
            and not np.allclose(components[index], want_components[index], atol=ATOL, rtol=RTOL)
        ]
        if wrong_sign:
            raise AssertionError(
                f"component {wrong_sign[0]} has the wrong sign. The entry with the largest absolute value "
                "must be positive; the lowest index breaks a tie.")
        plain, _ = _fit(case["X"], case["k"], center=False)
        if plain.shape == want_components.shape and np.allclose(components, plain, atol=ATOL, rtol=RTOL):
            if not np.allclose(components, want_components, atol=ATOL, rtol=RTOL):
                raise AssertionError("these are the components of the raw matrix. Subtract the column means first.")
    if ratio.shape == want_ratio.shape and want_ratio.sum() < 1.0 - 1e-6:
        renormalised = want_ratio / want_ratio.sum()
        if np.allclose(ratio, renormalised, atol=ATOL, rtol=RTOL) and not np.allclose(ratio, want_ratio, atol=ATOL, rtol=RTOL):
            raise AssertionError(
                "explained_variance_ratio was renormalised over the k returned components. "
                "Divide by the total variance of every component.")
    default_compare(got[0], want_components, ATOL, RTOL, "components")
    default_compare(got[1], want_ratio, ATOL, RTOL, "explained_variance_ratio")


def _random(seed, n, d, scales, k, name):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, d)) * np.asarray(scales, dtype=np.float64)
    data += rng.normal(size=d) * 4.0
    return {"_name": name, "X": data.tolist(), "k": k}


def _signed_cloud():
    """Both SVD and eigh return a leading component whose max-abs entry is negative."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(15, 4))
    data[:, 0] += 2.0
    return data.tolist()


def functional_cases():
    tilted = [[0.0, 10.0], [1.0, 10.0], [2.0, 10.0], [3.0, 11.0]]
    return [
        {"_name": "example", "X": tilted, "k": 1},
        {"_name": "both components, variance order", "X": tilted, "k": 2},
        {"_name": "center before decomposing", "X": [[100.0, 0.0], [100.0, 1.0], [100.0, 2.0]], "k": 1},
        {"_name": "sign of the leading component", "X": _signed_cloud(), "k": 2},
        {"_name": "one feature", "X": [[-2.0], [4.0]], "k": 1},
        {"_name": "two points in 3D", "X": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], "k": 1},
        _random(3, 40, 5, [5.0, 2.0, 0.8, 0.3, 0.1], 3, "random 40x5, k=3"),
        _random(4, 25, 6, [4.0, 1.5, 0.7, 0.25, 0.1, 0.04], 2, "random 25x6, k=2"),
        {"_name": "k = 0", "X": tilted, "k": 0},
        {"_name": "k larger than D", "X": tilted, "k": 3},
        {"_name": "k larger than N-1", "X": [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 0.0], [2.0, 2.0, 0.0, 1.0]], "k": 3},
        {"_name": "every row is equal", "X": [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], "k": 1},
    ]


def build() -> Problem:
    return Problem(
        slug="pca",
        title="Principal component analysis",
        difficulty="medium",
        summary="Top components of centered data, ordered by explained-variance ratio, with a fixed sign.",
        params=["X", "k"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "linear algebra", "dimensionality reduction"],
    )
