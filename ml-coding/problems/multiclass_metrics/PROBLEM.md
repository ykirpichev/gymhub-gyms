# Multiclass metrics and macro averaging

Write `solve(y_true, y_pred, num_classes)` for a single-label classifier with `K = num_classes` classes.

- `y_true`, `y_pred`: `N` integer labels in `[0, K)`.
- `num_classes`: `K ≥ 1`. Some classes may not occur in either list.

Return a dict:

- `"confusion"`: a `K × K` integer array; entry `[i, j]` counts examples whose **true class is `i` and predicted class is `j`** (rows are truth).
- `"precision"`, `"recall"`, `"f1"`: length-`K` float arrays, one value per class (one-vs-rest). `f1 = 2·P·R / (P + R)`. **Any metric with a zero denominator is `0.0`.**
- `"macro_precision"`, `"macro_recall"`, `"macro_f1"`: the unweighted mean of the per-class values over the **classes that appear in `y_true` or `y_pred`**. A class absent from both lists is left out of the mean; a class that appears in only one of them is included (with its 0.0). `macro_f1` is the mean of the per-class F1 values, not the F1 of the macro precision and recall.
- `"micro_precision"`, `"micro_recall"`, `"micro_f1"`: computed from TP, FP and FN summed over all classes. Python floats.

Raise `ValueError` if `N == 0`, the lists differ in length, or a label is outside `[0, K)`.

```
solve([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 0, 2], 4)
-> confusion  [[1, 1, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]
   precision  [0.5, 0.6667, 1.0, 0.0]   recall [0.5, 1.0, 0.5, 0.0]   f1 [0.5, 0.8, 0.6667, 0.0]
   macro_precision 0.7222   macro_recall 0.6667   macro_f1 0.6556   (class 3 is absent: mean over 3 classes)
   micro_precision = micro_recall = micro_f1 = 0.6667
```

**Interview focus:** per-class counts from the confusion matrix, macro versus micro averaging, and absent classes.

**Follow-ups** to answer out loud once your solution passes:

- When would you report macro averages instead of micro?
- Your dataset has 1,000 classes and a long tail. Which metrics do you show?
- How would you evaluate a multi-label problem?
