# Feature standardization with training statistics

Write `solve(train, test)`.

- `train`: `N × D` list of lists of numbers, `N >= 1`, `D >= 1`.
- `test`: `M × D` list of lists of numbers, `M >= 0`. Same `D` as train.

Return `{"train": [...], "test": [...]}` with both matrices standardized. For each column, the mean and the **population** standard deviation (`ddof = 0`, divide by `N`) come from `train` only:

`z = (x - mean) / std`

If that training standard deviation is 0, every train and test entry in the column is `0` (do not divide, and do not keep the centered test value). A test row must not change the mean or the standard deviation.

Raise `ValueError` if `train` is empty, a row has a different length from the others, or a cell is not a number.

```
solve([[1.0, 2.0], [3.0, 2.0]], [[5.0, 9.0]])
-> {"train": [[-1.0, 0.0], [1.0, 0.0]], "test": [[3.0, 0.0]]}
```

Column 0 has mean 2 and population std 1. Column 1 is constant, so the test value 9 becomes 0.

**Interview focus:** Fit/transform separation, constant features, shape validation.

**Follow-ups** to answer out loud once your solution passes:

- Why fit the scaler on training data only?
- Which models need standardized features, and which don't care?
- How would you standardize a feature with heavy outliers? (Robust scaling with the median and IQR.)
