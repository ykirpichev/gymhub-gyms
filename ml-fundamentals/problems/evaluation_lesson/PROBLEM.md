# Measuring a classifier

A binary classifier's predictions fall into four cells:

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | true positive (TP) | false negative (FN) |
| **Actually negative** | false positive (FP) | true negative (TN) |

Every common metric is a ratio of these cells:

- **Precision** $= TP / (TP + FP)$: of what we flagged, how much was right.
- **Recall** (sensitivity, true positive rate) $= TP / (TP + FN)$: of what was there, how much we found.
- **Specificity** $= TN / (TN + FP)$; the **false positive rate** is $1 - $ specificity.
- **F1** $= 2PR / (P + R)$, the harmonic mean, which is low if either is low.
- **Accuracy** $= (TP + TN) / N$, which is misleading when classes are imbalanced: predicting "negative" for a 1% positive class scores 99%.

Most classifiers output a score, and a **threshold** turns it into a decision. Sweeping the threshold traces curves:

- The **ROC curve** plots the true positive rate against the false positive rate. Its area (**ROC AUC**) is the probability that a random positive scores above a random negative; 0.5 is chance. It ignores class balance.
- The **precision-recall curve** is more informative when positives are rare, because precision collapses as false positives pile up, while the false positive rate barely moves.

For more than two classes, compute a metric per class and average: **macro** averages the per-class numbers equally, **micro** pools all the counts first (so frequent classes dominate).

**Model selection** needs data the model never trained on. **k-fold cross-validation** splits the training data into $k$ folds, trains on $k - 1$ and validates on the remaining one, $k$ times, and averages. Use **stratified** folds for imbalanced classes and **grouped** or time-based splits when rows are not independent (the same user, or a time series). Keep a test set aside and look at it once.

> [!WARNING]
> Choosing the threshold, features or hyperparameters on the test set leaks it into the model, and the reported score becomes optimistic.
