# Missing values and unseen categories

Write `solve(train, test, kinds)` for a table with missing cells and string categories.

- `train`: list of rows. Each row is a list of numbers, strings, or `None` (JSON `null`).
- `test`: rows in the same shape. It may be empty.
- `kinds`: one string per column, `"numeric"` or `"categorical"`.

Return `{"train": [...], "test": [...]}`. Both values are lists of float rows. Fit **only on `train`**. `test` is transformed with that state and must not change it.

Walk columns from left to right:

- **Numeric.** Replace `None` with the training median. Keep observed numbers. The median of an even count is the average of the two middle values. If the training column has no observed values, the fill is `0`.
- **Categorical.** Take the training strings, ignoring `None`. One-hot them in lexicographic order, then add one **unknown** column. Replace `None` with the mode before encoding. The mode is the most frequent string; ties go to the string that sorts first. A string that is not in the training set is unknown. If the training column has no strings, the only column is unknown and every row has `1` there.

Raise `ValueError` if a row length differs from `len(kinds)`, a kind is not `"numeric"` or `"categorical"`, a numeric cell is not a number or `None`, or a categorical cell is not a string or `None`.

```
solve(
  [[1, "b", None], [3, "a", 4], [None, "b", 8]],
  [[None, "c", None], [5, None, 1]],
  ["numeric", "categorical", "numeric"],
)
-> {
  "train": [[1.0, 0.0, 1.0, 0.0, 6.0], [3.0, 1.0, 0.0, 0.0, 4.0], [2.0, 0.0, 1.0, 0.0, 8.0]],
  "test": [[2.0, 0.0, 0.0, 1.0, 6.0], [5.0, 0.0, 1.0, 0.0, 1.0]],
}
```

Column 0 median is 2 and column 2 median is 6. Categories sort as `"a"`, `"b"`, then unknown; the missing category becomes `"b"`.

**Interview focus:** Training-only state, unknown bucket, all-missing columns.

**Follow-ups** to answer out loud once your solution passes:

- A category has a million distinct values. What do you do instead of one-hot? (Hashing, embeddings, target encoding.)
- How does target encoding leak, and how do you prevent it?
- Which models need one-hot encoding and which don't?
