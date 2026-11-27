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
