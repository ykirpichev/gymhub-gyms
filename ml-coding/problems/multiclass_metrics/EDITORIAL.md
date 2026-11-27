#### The idea
Build the confusion matrix with one scatter-add, `np.add.at(cm, (t, p), 1)`. Then every one-vs-rest metric is a
vector expression over it: the diagonal is TP, column sums are "predicted as c" (TP + FP), row sums are "truly c"
(TP + FN). `np.divide(..., where=denominator > 0)` handles the zero cases without warnings.

Macro averages weight every class equally, so rare classes matter. Micro averages pool the counts, so frequent classes
dominate. In single-label classification every mistake is one FP (for the predicted class) and one FN (for the true
one), so micro precision, micro recall and micro F1 all equal accuracy.

#### A solution
```python
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
```

#### What it costs
`O(N + K²)`. The confusion matrix is the whole state, so metrics over a huge evaluation set can be computed in streaming
fashion by summing per-shard matrices.

#### Pitfalls
- `cm[t, p] += 1` with fancy indexing: repeated pairs are counted once. That is why `np.add.at` exists.
- Rows and columns swapped, so precision and recall swap.
- Macro F1 computed as the F1 of macro precision and macro recall.
- Including classes absent from both lists in the macro mean, which drags it toward 0.
