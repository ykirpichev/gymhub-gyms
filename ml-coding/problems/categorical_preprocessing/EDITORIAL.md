#### The idea
Fit a small **plan** per column from training rows only, then encode every row (train and test) with those plans:

- numeric: the median of the observed values (robust to outliers), used to fill `None`;
- categorical: the sorted list of training categories plus one "unknown" slot, and the mode to fill `None`.

Because the plans are frozen before any test row is seen, a test value cannot change the encoding, and a category the
model never trained on lands in the unknown column instead of crashing.

#### A solution
```python
from collections import Counter


def solve(train, test, kinds):
    _validate(train, kinds, "train")
    _validate(test, kinds, "test")
    plans = [_fit_column(train, kinds, j) for j in range(len(kinds))]
    return {"train": [_encode(row, plans) for row in train], "test": [_encode(row, plans) for row in test]}


def _validate(rows, kinds, name):
    for row in rows:
        if len(row) != len(kinds):
            raise ValueError(f"{name} row has {len(row)} columns, expected {len(kinds)}")
        for value, kind in zip(row, kinds):
            if kind == "numeric":
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError("numeric cells must be numbers or null")
            elif kind == "categorical":
                if value is not None and not isinstance(value, str):
                    raise ValueError("categorical cells must be strings or null")
            else:
                raise ValueError(f"unknown kind {kind!r}")


def _fit_column(train, kinds, index):
    observed = [row[index] for row in train if row[index] is not None]
    if kinds[index] == "numeric":
        return ("numeric", _median(observed))
    if not observed:
        return ("categorical", None, [])
    return ("categorical", _mode(observed), sorted(set(observed)))


def _median(values):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def _mode(values):
    counts = Counter(values)
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def _encode(row, plans):
    encoded = []
    for value, plan in zip(row, plans):
        if plan[0] == "numeric":
            encoded.append(float(plan[1] if value is None else value))
            continue
        _, mode, categories = plan
        if value is None:
            value = mode
        one_hot = [0.0] * (len(categories) + 1)
        try:
            one_hot[categories.index(value)] = 1.0
        except ValueError:
            one_hot[-1] = 1.0
        encoded.extend(one_hot)
    return encoded
```

#### What it costs
Fitting is `O(N·D)` plus a sort per column. The one-hot width grows with the number of categories; for high-cardinality
features (user IDs, URLs) you switch to hashing (a fixed number of buckets) or learned embeddings, and you often group
rare categories into the unknown bucket so it gets trained.

#### Pitfalls
- Fitting the imputer or the vocabulary on `train + test`.
- Ordering categories by first appearance instead of sorting, so the column order depends on the data order.
- Filling a missing category with the unknown slot instead of the mode (a legitimate choice, but not this spec).
- No unknown column at all: the first unseen category in production raises a `KeyError`.
- Treating `True`/`False` as numbers, since `bool` is a subclass of `int` in Python.
