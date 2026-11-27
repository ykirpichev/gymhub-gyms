import numpy as np


def _average_ranks(x):
    """1-based ranks of x; tied values share the mean of the ranks they span."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start
        while end + 1 < len(x) and sorted_x[end + 1] == sorted_x[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def solve(y_true, scores, threshold):
    if len(y_true) == 0 or len(y_true) != len(scores):
        raise ValueError("y_true and scores must be non-empty and the same length")
    y = np.asarray(y_true)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    s = np.asarray(scores, dtype=np.float64)
    positive = y == 1
    pred = s >= threshold

    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = float(np.mean(pred == positive))

    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    auc = None
    if n_pos and n_neg:
        # Mann-Whitney U: rank sum of the positives minus its minimum; average ranks give ties 1/2.
        rank_sum = _average_ranks(s)[positive].sum()
        auc = float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "auc": auc}
