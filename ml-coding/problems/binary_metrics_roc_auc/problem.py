"""Grader for binary_metrics_roc_auc. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

KEYS = ("precision", "recall", "f1", "accuracy", "auc")


def reference(y_true, scores, threshold):
    if len(y_true) == 0 or len(y_true) != len(scores) or any(y not in (0, 1) for y in y_true):
        return "ValueError"
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float(np.mean(pred == (y == 1)))
    pos, neg = s[y == 1], s[y == 0]
    auc = None
    if len(pos) and len(neg):
        # Brute force over every (positive, negative) pair: an independent check of rank-based solutions.
        diff = pos[:, None] - neg[None, :]
        auc = float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "accuracy": accuracy, "auc": auc}


def call(module, args):
    """Invalid input must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError("expected ValueError (empty input, length mismatch or a label outside {0, 1})"
                                 if want == "ValueError" else "raised ValueError, but the input is valid")
        return
    if not isinstance(got, dict):
        raise AssertionError(f"return a dict with keys {list(KEYS)}, got {type(got).__name__}")
    missing = [key for key in KEYS if key not in got]
    if missing:
        raise AssertionError(f"the result dict is missing {missing}")
    for key in KEYS:
        if want[key] is None or got[key] is None:
            if got[key] is not want[key]:
                raise AssertionError("auc: expected None, only one class is present" if want[key] is None
                                     else f"{key}: got None, expected {want[key]!r}")
            continue
        default_compare(got[key], want[key], 1e-9, 1e-9, key)


def _random(seed, n, decimals, name, threshold=0.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # Positives score higher on average; rounding creates many ties across classes.
    scores = np.round(np.clip(rng.normal(0.4 + 0.2 * y, 0.2), 0, 1), decimals)
    return {"_name": name, "y_true": y.tolist(), "scores": scores.tolist(), "threshold": threshold}


def functional_cases():
    return [
        {"_name": "example", "y_true": [0, 0, 1, 1, 1, 0], "scores": [0.1, 0.4, 0.35, 0.8, 0.4, 0.2], "threshold": 0.4},
        {"_name": "score equal to threshold is positive", "y_true": [1, 0], "scores": [0.5, 0.5], "threshold": 0.5},
        {"_name": "all scores tied", "y_true": [1, 0, 1, 0, 0], "scores": [0.3] * 5, "threshold": 0.9},
        {"_name": "perfect ranking", "y_true": [0, 0, 1, 1], "scores": [0.1, 0.2, 0.8, 0.9], "threshold": 0.5},
        {"_name": "reversed ranking", "y_true": [1, 1, 0, 0], "scores": [-2.0, -1.0, 3.0, 4.0], "threshold": 0.0},
        {"_name": "no predicted positives", "y_true": [1, 0, 1], "scores": [0.2, 0.1, 0.3], "threshold": 0.99},
        {"_name": "only positives: AUC undefined", "y_true": [1, 1, 1], "scores": [0.9, 0.2, 0.6], "threshold": 0.5},
        {"_name": "only negatives: everything 0 but accuracy", "y_true": [0, 0], "scores": [0.1, 0.7], "threshold": 0.5},
        _random(1, 40, 1, "random 40, heavy ties"),
        _random(2, 3000, 2, "random 3000, ties", threshold=0.45),
        {"_name": "label 2 is invalid", "y_true": [0, 2, 1], "scores": [0.1, 0.2, 0.3], "threshold": 0.5},
        {"_name": "length mismatch", "y_true": [0, 1], "scores": [0.1, 0.2, 0.3], "threshold": 0.5},
        {"_name": "empty input", "y_true": [], "scores": [], "threshold": 0.5},
    ]


def build() -> Problem:
    return Problem(
        slug="binary_metrics_roc_auc",
        title="Binary metrics and ROC-AUC",
        difficulty="medium",
        summary="Precision, recall, F1, accuracy at a threshold, plus ROC-AUC with tied scores counted as half.",
        params=["y_true", "scores", "threshold"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["evaluation", "ranking", "class imbalance"],
    )
