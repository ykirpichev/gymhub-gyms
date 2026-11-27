"""Grader for standardize_features. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Mean and population std are fit on train only. A column with training std 0 transforms to 0,
including test rows that are not equal to that constant. Concatenating train and test before
the fit fails the leakage cases.
"""

import math

import numpy as np

from gymkit.spec import Problem


def _invalid(train, test):
    if len(train) == 0 or len(train[0]) == 0:
        return "empty train"
    width = len(train[0])
    for name, rows in (("train", train), ("test", test)):
        for row in rows:
            if len(row) != width:
                return f"{name} feature count does not match"
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return f"{name} cell is not a number"
    return None


def reference(train, test):
    if _invalid(train, test):
        return "ValueError"
    width = len(train[0])
    n = len(train)
    means = []
    stds = []
    for j in range(width):
        column = [float(row[j]) for row in train]
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
                    out.append((float(value) - means[j]) / stds[j])
            scaled.append(out)
        return scaled

    return {"train": apply(train), "test": apply(test)}


def call(module, args):
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError for an empty train, a ragged row, or a non-numeric cell"
                if want == "ValueError" else "raised ValueError, but the matrices are valid"
            )
        return
    if not isinstance(got, dict) or not {"train", "test"} <= set(got):
        raise AssertionError("return a dict with keys 'train' and 'test'")
    try:
        default_compare(got["train"], want["train"], 1e-8, 1e-6, "train")
        default_compare(got["test"], want["test"], 1e-8, 1e-6, "test")
    except AssertionError as exc:
        raise AssertionError(
            str(exc) + "\n  fit the mean and population std on the training rows only; std 0 maps every row to 0"
        ) from None


def _random_case(seed, n, m, d, name):
    rng = np.random.default_rng(seed)
    train = rng.normal(loc=1.5, scale=2.0, size=(n, d)).tolist()
    test = rng.normal(loc=-0.5, scale=3.0, size=(m, d)).tolist()
    for row in train:
        row[0] = 4.0
    return {"_name": name, "train": train, "test": test}


def functional_cases():
    return [
        {"_name": "example", "train": [[1.0, 2.0], [3.0, 2.0]], "test": [[5.0, 9.0]]},
        {"_name": "population std, not sample std", "train": [[1.0], [2.0], [3.0], [4.0]], "test": [[2.5]]},
        {"_name": "single training row is constant", "train": [[4.0, -1.0]], "test": [[8.0, 3.0]]},
        {"_name": "empty test", "train": [[1.0], [3.0]], "test": []},
        {"_name": "negative values, constant second column", "train": [[-2.0, 4.0], [2.0, 4.0]], "test": [[0.0, 9.0]]},
        {"_name": "leak: constant training column", "train": [[0.0], [0.0], [0.0]], "test": [[4.0]]},
        {"_name": "leak: test row must not shift the mean", "train": [[1.0], [3.0]], "test": [[9.0]]},
        _random_case(3, 6, 4, 3, "random 6x3 with a constant column"),
        _random_case(4, 40, 15, 8, "random 40x8 with a constant column"),
        {"_name": "mismatched feature count", "train": [[1.0, 2.0], [3.0, 4.0]], "test": [[5.0]]},
        {"_name": "ragged training row", "train": [[1.0, 2.0], [3.0]], "test": [[4.0, 5.0]]},
        {"_name": "empty train", "train": [], "test": [[1.0, 2.0]]},
        {"_name": "non-numeric cell", "train": [[1.0, "x"]], "test": [[2.0, 3.0]]},
    ]


def build() -> Problem:
    return Problem(
        slug="standardize_features",
        title="Feature standardization with training statistics",
        difficulty="easy",
        summary="Standardize features with the training mean and population standard deviation, including constant columns.",
        params=["train", "test"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["preprocessing", "standardization", "leakage"],
    )
