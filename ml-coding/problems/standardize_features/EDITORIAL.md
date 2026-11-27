#### The idea
Standardisation has two phases, **fit** and **transform**, and only training data goes into fit. Compute each column's
mean and population standard deviation on `train`, then apply the same `(x − mean)/std` to both matrices. Test rows are
transformed with numbers they did not influence, exactly as production traffic will be.

#### A solution
```python
import math


def solve(train, test):
    if len(train) == 0:
        raise ValueError("train must contain at least one row")
    width = len(train[0])
    if width == 0:
        raise ValueError("train must contain at least one feature")
    train_m = _matrix(train, width, "train")
    test_m = _matrix(test, width, "test")
    n = len(train_m)
    means = []
    stds = []
    for j in range(width):
        column = [row[j] for row in train_m]
        mean = sum(column) / n
        variance = sum((value - mean) ** 2 for value in column) / n
        means.append(mean)
        stds.append(math.sqrt(variance))

    def apply(rows):
        scaled = []
        for row in rows:
            out = []
            for j, value in enumerate(row):
                if stds[j] == 0.0:
                    out.append(0.0)
                else:
                    out.append((value - means[j]) / stds[j])
            scaled.append(out)
        return scaled

    return {"train": apply(train_m), "test": apply(test_m)}


def _matrix(rows, width, name):
    parsed = []
    for row in rows:
        if len(row) != width:
            raise ValueError(f"{name} row has {len(row)} features, expected {width}")
        try:
            parsed.append([float(value) for value in row])
        except (TypeError, ValueError):
            raise ValueError(f"{name} features must be numbers") from None
    return parsed
```

A NumPy version is shorter (`train.mean(axis=0)`, `train.std(axis=0)`, `np.where(std == 0, 0, ...)`), but the plain
loops make the fit/transform split explicit.

#### What it costs
`O((N + M)·D)`. In production, the fitted means and stds are part of the model artifact and are saved next to the
weights; recomputing them on serving traffic is a classic training/serving skew bug.

#### Pitfalls
- Computing statistics on `train + test`, which leaks test information into training.
- Sample std (`ddof = 1`) when the spec asks for population std.
- Dividing a constant column by 0, or leaving the centered test value instead of 0.
- Modifying the input lists in place.
