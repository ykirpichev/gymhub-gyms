# Binary metrics and ROC-AUC

Write `solve(y_true, scores, threshold)` for a binary classifier evaluated on `N` examples.

- `y_true`: `N` integer labels, each `0` or `1`.
- `scores`: `N` floats; a higher score means "more likely positive".
- `threshold`: a float. An example is **predicted positive when `score >= threshold`**.

Return a dict with five keys:

- `"precision"` = TP / (TP + FP), `"recall"` = TP / (TP + FN), `"f1"` = 2·P·R / (P + R), `"accuracy"` = correct / N. Each is a Python float; **when a denominator is 0 the metric is `0.0`**.
- `"auc"`: the ROC-AUC, the probability that a random positive scores higher than a random negative. Over all (positive, negative) pairs, a pair counts **1** if the positive's score is higher, **1/2 if the scores are tied**, and 0 otherwise; divide by (#positives · #negatives). The threshold plays no part. **If `y_true` contains only one class, `"auc"` is `None`.**

Raise `ValueError` if `N == 0`, if the two lists have different lengths, or if a label is not 0 or 1.

```
solve([0, 0, 1, 1, 1, 0], [0.1, 0.4, 0.35, 0.8, 0.4, 0.2], 0.4)
-> {"precision": 0.6667, "recall": 0.6667, "f1": 0.6667, "accuracy": 0.6667, "auc": 0.8333}
```

The positive scored 0.4 ties the negative scored 0.4: that pair adds 1/2, so AUC = 7.5 / 9.

**Interview focus:** zero denominators, tied scores, and AUC as a rank statistic (Mann-Whitney U) computed in O(N log N) rather than over all pairs.

**Follow-ups** to answer out loud once your solution passes:

- How would you pick a decision threshold for a fraud model where a missed fraud costs 50 times a false alarm?
- When is PR AUC more informative than ROC AUC?
- Your AUC is 0.99 offline and the model fails in production. What do you suspect first?
