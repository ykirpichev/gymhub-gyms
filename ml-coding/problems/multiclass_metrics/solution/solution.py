import numpy as np


def solve(y_true, y_pred, num_classes):
    t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_pred, dtype=np.int64)
    k = num_classes
    if t.size == 0 or t.shape != p.shape:
        raise ValueError("y_true and y_pred must be non-empty and the same length")
    if t.min() < 0 or p.min() < 0 or t.max() >= k or p.max() >= k:
        raise ValueError("labels must be in [0, num_classes)")

    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (t, p), 1)

    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0)  # column sums: how often each class was predicted
    actual = cm.sum(axis=1)     # row sums: how often each class is the truth
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(k), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)

    seen = (predicted + actual) > 0  # classes absent from both lists do not count in macro averages
    micro = float(tp.sum() / t.size)  # single-label: every error is one FP and one FN, so all equal accuracy
    return {
        "confusion": cm,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision[seen].mean()),
        "macro_recall": float(recall[seen].mean()),
        "macro_f1": float(f1[seen].mean()),
        "micro_precision": micro,
        "micro_recall": micro,
        "micro_f1": micro,
    }
