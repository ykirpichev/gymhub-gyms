# Train/validation/test split without leakage

Write `solve(n, fractions, labels, seed)`.

Split indices `0 .. n-1` into train, validation and test.

- `n`: number of rows, an int, `n >= 0`.
- `fractions`: `[train, validation, test]`, three numbers `>= 0`. They do not have to sum to 1.
- `labels`: `None`, or `n` labels (all ints or all strings) to stratify.
- `seed`: an int. Shuffle with `random.Random(seed)`.

Return `{"train": [...], "validation": [...], "test": [...]}` — three lists of indices. Order inside a list does not matter. The lists are disjoint and their union is every index.

**Sizes.** `sizes[i] = math.floor(n * fractions[i] + 1e-12)`. Let `leftover = n - sum(sizes)`. Give those leftover slots one at a time to train, then validation, then test, cycling, so the lengths sum to `n`. For `n = 10` and `[0.33, 0.33, 0.34]` the lengths are `4, 3, 3`.

Raise `ValueError` if `n < 0`, `fractions` does not have length 3, a fraction is negative, `leftover < 0`, or `labels` is not `None` and `len(labels) != n`. With labels, also raise `ValueError` when the split sizes cannot satisfy the per-class rules below: for example `n = 4`, `[1, 0, 0]` and labels `[0, 0, 0, 1]`, where a class of 3 needs an index in validation and in test but those splits have no slots.

**No labels.** Shuffle `range(n)` and cut it into the three lengths. The same seed returns the same split. Every index is equally likely to land in a given split.

**Stratified.** Shuffle inside each class. For a class of size `c`, the three counts sum to `c`, and split `s` is within ±1 of `c * sizes[s] / n`. A class of 1 goes entirely to train. A class of 2 puts one index in train and one in validation. A class of 3 or more puts at least one index in every split.

```
solve(9, [1/3, 1/3, 1/3], [0, 0, 0, 1, 1, 1, 2, 2, 2], 1)
-> lengths 3, 3 and 3. Each split contains one index of each label.
```

Which index of a class lands in train depends on the seed. The grader checks disjointness, the sizes, the per-class rules, that the same seed repeats, and that members of a class are spread by the shuffle. It does not expect one particular permutation.

**Interview focus:** Disjoint indices, class balance, and a reproducible split so a later fit uses only the training rows.

**Follow-ups** to answer out loud once your solution passes:

- The same user appears in many rows. How do you split now?
- How do you split a time series?
- Why stratify, and when can stratification fail?
