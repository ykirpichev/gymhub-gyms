## Hint 1
Build the `M × N` matrix of squared distances in one go. `‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b`, and `X_test @ X_train.T` gives all the dot products. You do not need the square root to rank.

## Hint 2
`np.argsort(d2, axis=1, kind="stable")` orders each row by distance and keeps the lower index first among equal values. `np.argpartition` is faster but gives no guarantee about which of several equal distances it keeps.

## Hint 3
`y_train[nearest]` is an `M × k` array of labels. For each row, `np.bincount(row, minlength=n_labels).argmax()` gives the most frequent label, and `argmax` returns the first (smallest) label when counts tie.
