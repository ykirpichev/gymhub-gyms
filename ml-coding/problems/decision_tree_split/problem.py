"""Grader for decision_tree_split. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

TIE_TOL = 1e-12


def _gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return float(1.0 - np.sum(proportions ** 2))


def reference(X, y):
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        return "ValueError"
    if features.shape[0] == 0 or features.shape[1] == 0:
        return "ValueError"
    labels = labels.astype(np.int64, copy=False)
    parent = _gini(labels)
    best = None
    for feature in range(features.shape[1]):
        values = np.unique(features[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2.0:
            left = features[:, feature] <= threshold
            impurity = (left.sum() * _gini(labels[left]) + (~left).sum() * _gini(labels[~left])) / len(labels)
            candidate = (float(impurity), feature, float(threshold))
            if best is None or candidate[0] < best[0] - TIE_TOL:
                best = candidate
    if best is None or best[0] >= parent - TIE_TOL:
        return None
    return best[1], best[2], best[0]


def _unweighted(X, y):
    """Best split when each side is averaged, not weighted by its size. None if no threshold exists."""
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y).astype(np.int64, copy=False)
    best = None
    for feature in range(features.shape[1]):
        values = np.unique(features[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2.0:
            left = features[:, feature] <= threshold
            score = 0.5 * (_gini(labels[left]) + _gini(labels[~left]))
            candidate = (float(score), feature, float(threshold))
            if best is None or candidate[0] < best[0] - TIE_TOL:
                best = candidate
    return None if best is None else (best[1], best[2])


def call(module, args):
    """A row count that does not match y must raise ValueError."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError: X is N x D and y has N integer labels"
                if want == "ValueError" else "raised ValueError, but X and y are valid")
        return
    if want is None or got is None:
        if got is not None or want is not None:
            raise AssertionError(
                "expected None: no candidate has a weighted Gini strictly below the parent "
                "(one row, constant features, equal labels, or a split that does not change the mix)"
                if want is None else "returned None, but a split improves on the parent Gini")
        return
    if not isinstance(got, tuple) or len(got) != 3:
        raise AssertionError(f"return (feature, threshold, impurity) or None, got {type(got).__name__}")
    feature = int(got[0]) if isinstance(got[0], (int, np.integer)) and not isinstance(got[0], bool) else got[0]
    if feature != want[0]:
        raise AssertionError(
            f"feature: got {got[0]!r}, expected {want[0]}. "
            "Minimise the weighted Gini; on a tie take the lowest feature index, then the lowest threshold.")
    threshold = float(got[1]) if isinstance(got[1], (int, float, np.floating)) else None
    if threshold is None or not np.isclose(threshold, want[1], atol=1e-9, rtol=0):
        column = np.asarray(case["X"], dtype=np.float64)
        landed_on_a_value = (
            isinstance(feature, (int, np.integer)) and 0 <= feature < column.shape[1]
            and threshold is not None and np.any(np.isclose(np.unique(column[:, feature]), threshold, atol=1e-9)))
        if landed_on_a_value:
            raise AssertionError(
                f"threshold: got {threshold:.6g}, expected {want[1]:.6g}. "
                "Use the midpoint between consecutive distinct sorted values, not a data value.")
        other = _unweighted(case["X"], case["y"])
        if other is not None and feature == other[0] and threshold is not None and np.isclose(threshold, other[1], atol=1e-9):
            raise AssertionError(
                f"threshold: got {threshold:.6g}, expected {want[1]:.6g}. "
                "Weight each side by its number of rows, (n_left * gini_left + n_right * gini_right) / n.")
        raise AssertionError(
            f"threshold: got {got[1]!r}, expected {want[1]:.6g}. "
            "On a tie, choose the lowest feature index and then the lowest threshold.")
    if isinstance(got[2], (int, float, np.floating)):
        labels = np.asarray(case["y"])
        _, counts = np.unique(labels, return_counts=True)
        parent = 1.0 - np.sum((counts / counts.sum()) ** 2)
        if np.isclose(float(got[2]), parent - want[2], atol=1e-8) and not np.isclose(float(got[2]), want[2], atol=1e-8):
            raise AssertionError(
                f"impurity: got {float(got[2]):.6g}, expected {want[2]:.6g}. "
                "Return the weighted Gini of the split, not how much it improves on the parent.")
    default_compare(got[2], want[2], 1e-9, 1e-8, "weighted Gini impurity")


def _random(seed, n, d, n_classes, name):
    rng = np.random.default_rng(seed)
    return {
        "_name": name,
        "X": rng.integers(0, 6, size=(n, d)).astype(float).tolist(),
        "y": rng.integers(0, n_classes, size=n).tolist(),
    }


def functional_cases():
    return [
        {"_name": "example", "X": [[1.0, 2.0], [2.0, 1.0], [3.0, 2.0], [4.0, 1.0]], "y": [0, 0, 1, 1]},
        {"_name": "tie: lowest threshold", "X": [[0.0], [1.0], [2.0]], "y": [0, 1, 0]},
        {"_name": "tie: lowest feature", "X": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], "y": [0, 1, 0]},
        {"_name": "duplicates: one midpoint, rows unsorted",
         "X": [[3.0], [1.0], [1.0], [3.0], [1.0]], "y": [1, 0, 1, 1, 0]},
        {"_name": "all labels equal", "X": [[1.0], [2.0], [3.0], [4.0]], "y": [2, 2, 2, 2]},
        {"_name": "constant features", "X": [[5.0, 7.0], [5.0, 7.0], [5.0, 7.0]], "y": [0, 1, 0]},
        {"_name": "single row", "X": [[1.0, 2.0]], "y": [0]},
        {"_name": "no split improves", "X": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], "y": [0, 1, 1, 0]},
        {"_name": "weight each side by its size",
         "X": [[4.0, 3.0], [2.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 4.0], [3.0, 4.0], [2.0, 3.0], [4.0, 3.0]],
         "y": [1, 1, 1, 1, 0, 1, 1, 0]},
        {"_name": "labels are not 0 and 1", "X": [[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [3.0, 1.0]], "y": [-1, -1, 4, 4]},
        {"_name": "multi-class tie: lowest threshold",
         "X": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], "y": [0, 0, 1, 1, 2, 2]},
        _random(1, 30, 4, 3, "random 30x4, 3 classes"),
        _random(2, 80, 5, 4, "random 80x5, 4 classes"),
        {"_name": "y length mismatch", "X": [[0.0], [1.0], [2.0]], "y": [0, 1]},
    ]


def build() -> Problem:
    return Problem(
        slug="decision_tree_split",
        title="Decision-tree split",
        difficulty="medium",
        summary="The numeric split with the lowest weighted Gini, using midpoint thresholds and a fixed tie-break.",
        params=["X", "y"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "trees", "gini"],
    )
