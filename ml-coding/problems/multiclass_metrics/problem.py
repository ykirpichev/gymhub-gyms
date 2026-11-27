"""Grader for multiclass_metrics. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

KEYS = ("confusion", "precision", "recall", "f1", "macro_precision", "macro_recall", "macro_f1",
        "micro_precision", "micro_recall", "micro_f1")


def _safe(num, den):
    return num / den if den else 0.0


def reference(y_true, y_pred, num_classes):
    k = num_classes
    if len(y_true) == 0 or len(y_true) != len(y_pred) or any(not 0 <= v < k for v in list(y_true) + list(y_pred)):
        return "ValueError"
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    precision, recall, f1 = [], [], []
    for c in range(k):
        tp, predicted, actual = cm[c, c], cm[:, c].sum(), cm[c, :].sum()
        pr, rc = _safe(tp, predicted), _safe(tp, actual)
        precision.append(float(pr))
        recall.append(float(rc))
        f1.append(float(_safe(2 * pr * rc, pr + rc)))
    seen = [c for c in range(k) if cm[c, :].sum() + cm[:, c].sum() > 0]
    micro = float(np.trace(cm) / cm.sum())
    return {
        "confusion": cm,
        "precision": np.array(precision),
        "recall": np.array(recall),
        "f1": np.array(f1),
        "macro_precision": float(np.mean([precision[c] for c in seen])),
        "macro_recall": float(np.mean([recall[c] for c in seen])),
        "macro_f1": float(np.mean([f1[c] for c in seen])),
        "micro_precision": micro,
        "micro_recall": micro,
        "micro_f1": micro,
    }


def call(module, args):
    """Invalid input must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if not (isinstance(got, str) and got == want):
            raise AssertionError("expected ValueError (empty input, length mismatch or a label outside [0, K))"
                                 if isinstance(want, str) else "raised ValueError, but the input is valid")
        return
    if not isinstance(got, dict):
        raise AssertionError(f"return a dict with keys {list(KEYS)}, got {type(got).__name__}")
    missing = [key for key in KEYS if key not in got]
    if missing:
        raise AssertionError(f"the result dict is missing {missing}")
    for key in KEYS:
        default_compare(got[key], want[key], 1e-9, 1e-9, key)


def _random(seed, n, k, accuracy, name):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, n)
    noise = rng.integers(0, k, n)
    y_pred = np.where(rng.random(n) < accuracy, y_true, noise)
    return {"_name": name, "y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "num_classes": k}


def functional_cases():
    return [
        {"_name": "example", "y_true": [0, 0, 1, 1, 2, 2], "y_pred": [0, 1, 1, 1, 0, 2], "num_classes": 4},
        {"_name": "all correct", "y_true": [2, 0, 1, 0], "y_pred": [2, 0, 1, 0], "num_classes": 3},
        {"_name": "class never predicted", "y_true": [0, 1, 1, 2], "y_pred": [0, 0, 0, 2], "num_classes": 3},
        {"_name": "class predicted but never true", "y_true": [0, 0, 0, 1], "y_pred": [0, 2, 0, 1], "num_classes": 3},
        {"_name": "one class only, K=5", "y_true": [3, 3, 3], "y_pred": [3, 3, 3], "num_classes": 5},
        {"_name": "everything wrong", "y_true": [0, 1, 2], "y_pred": [1, 2, 0], "num_classes": 3},
        {"_name": "imbalanced: macro vs micro", "y_true": [0] * 9 + [1], "y_pred": [0] * 10, "num_classes": 2},
        _random(1, 50, 6, 0.6, "random 50 examples, K=6"),
        _random(2, 5000, 10, 0.7, "random 5000 examples, K=10"),
        {"_name": "label equal to K", "y_true": [0, 1], "y_pred": [0, 2], "num_classes": 2},
        {"_name": "negative label", "y_true": [-1, 1], "y_pred": [0, 1], "num_classes": 2},
        {"_name": "length mismatch", "y_true": [0, 1, 1], "y_pred": [0, 1], "num_classes": 2},
        {"_name": "empty input", "y_true": [], "y_pred": [], "num_classes": 3},
    ]


def build() -> Problem:
    return Problem(
        slug="multiclass_metrics",
        title="Multiclass metrics and macro averaging",
        difficulty="medium",
        summary="Confusion matrix, per-class precision, recall and F1, and their macro and micro averages over K classes.",
        params=["y_true", "y_pred", "num_classes"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["evaluation", "confusion matrix", "averaging"],
    )
