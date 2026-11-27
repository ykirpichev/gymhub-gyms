"""Grader for categorical_preprocessing. Private: learners see PROBLEM.md, HINTS.md and the starter only.

The reference fits medians, modes and category lists on the training rows only. Cases where the
test set would move a median or add a category fail a solution that concatenates train and test.
"""

from collections import Counter

import numpy as np

from gymkit.spec import Problem


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


def _invalid(train, test, kinds):
    if not isinstance(kinds, (list, tuple)) or any(kind not in ("numeric", "categorical") for kind in kinds):
        return "kinds must be numeric or categorical"
    for name, rows in (("train", train), ("test", test)):
        for row in rows:
            if len(row) != len(kinds):
                return f"{name} row width does not match kinds"
            for value, kind in zip(row, kinds):
                if kind == "numeric" and value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    return "numeric cell is not a number"
                if kind == "categorical" and value is not None and not isinstance(value, str):
                    return "categorical cell is not a string"
    return None


def reference(train, test, kinds):
    if _invalid(train, test, kinds):
        return "ValueError"
    plans = []
    for index, kind in enumerate(kinds):
        observed = [row[index] for row in train if row[index] is not None]
        if kind == "numeric":
            plans.append(("numeric", _median(observed)))
        elif not observed:
            plans.append(("categorical", None, []))
        else:
            plans.append(("categorical", _mode(observed), sorted(set(observed))))

    def encode(row):
        encoded = []
        for value, plan in zip(row, plans):
            if plan[0] == "numeric":
                encoded.append(float(plan[1] if value is None else value))
                continue
            _, mode, categories = plan
            if value is None:
                value = mode
            one_hot = [0.0] * (len(categories) + 1)
            if value in categories:
                one_hot[categories.index(value)] = 1.0
            else:
                one_hot[-1] = 1.0
            encoded.extend(one_hot)
        return encoded

    return {"train": [encode(row) for row in train], "test": [encode(row) for row in test]}


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
                "expected ValueError for a bad row or kind" if want == "ValueError"
                else "raised ValueError, but the table is valid"
            )
        return
    if not isinstance(got, dict) or not {"train", "test"} <= set(got):
        raise AssertionError("return a dict with keys 'train' and 'test'")
    try:
        default_compare(got["train"], want["train"], 1e-8, 1e-6, "train")
        default_compare(got["test"], want["test"], 1e-8, 1e-6, "test")
    except AssertionError as exc:
        raise AssertionError(
            str(exc) + "\n  fit the median, the mode and the category list on the training rows only"
        ) from None


def _random_case(seed, n_train, n_test, name):
    rng = np.random.default_rng(seed)
    kinds = ["numeric", "categorical", "numeric", "categorical"]
    colors = ["red", "green", "blue"]
    flags = ["x", "y"]
    train = []
    for _ in range(n_train):
        train.append([
            None if rng.random() < 0.25 else float(rng.normal(0, 2)),
            None if rng.random() < 0.2 else colors[int(rng.integers(0, 3))],
            None if rng.random() < 0.2 else float(rng.integers(-2, 5)),
            None if rng.random() < 0.2 else flags[int(rng.integers(0, 2))],
        ])
    test = []
    for _ in range(n_test):
        test.append([
            None if rng.random() < 0.3 else float(rng.normal(0, 2)),
            None if rng.random() < 0.25 else ["red", "green", "blue", "purple"][int(rng.integers(0, 4))],
            None if rng.random() < 0.3 else float(rng.integers(-2, 5)),
            None if rng.random() < 0.25 else ["x", "y", "z"][int(rng.integers(0, 3))],
        ])
    return {"_name": name, "train": train, "test": test, "kinds": kinds}


def functional_cases():
    return [
        {
            "_name": "example",
            "train": [[1, "b", None], [3, "a", 4], [None, "b", 8]],
            "test": [[None, "c", None], [5, None, 1]],
            "kinds": ["numeric", "categorical", "numeric"],
        },
        {
            "_name": "mode tie sorts first",
            "train": [[None, "b"], [None, "a"]],
            "test": [[1, None]],
            "kinds": ["numeric", "categorical"],
        },
        {
            "_name": "even median",
            "train": [[1], [2], [3], [4]],
            "test": [[None]],
            "kinds": ["numeric"],
        },
        {
            "_name": "all-missing numeric",
            "train": [[None, "x"], [None, "x"]],
            "test": [[4, "y"], [None, None]],
            "kinds": ["numeric", "categorical"],
        },
        {
            "_name": "all-missing categorical",
            "train": [[1, None], [5, None]],
            "test": [[None, "q"]],
            "kinds": ["numeric", "categorical"],
        },
        {
            "_name": "unknown column even if unused",
            "train": [["red"], ["blue"]],
            "test": [["red"]],
            "kinds": ["categorical"],
        },
        {
            "_name": "empty test",
            "train": [[1.0, "a"]],
            "test": [],
            "kinds": ["numeric", "categorical"],
        },
        {
            "_name": "empty train is all missing",
            "train": [],
            "test": [[None, "x"], [7, None]],
            "kinds": ["numeric", "categorical"],
        },
        {
            "_name": "two categorical columns",
            "train": [["a", "y"], ["b", "y"]],
            "test": [["a", "z"]],
            "kinds": ["categorical", "categorical"],
        },
        {
            "_name": "leak: test values must not move the median",
            "train": [[None], [0], [0]],
            "test": [[100], [100]],
            "kinds": ["numeric"],
        },
        {
            "_name": "leak: unseen test category stays unknown",
            "train": [["a"], ["a"]],
            "test": [["z"], ["z"]],
            "kinds": ["categorical"],
        },
        _random_case(7, 30, 12, "random mixed table"),
        _random_case(8, 8, 5, "random small table"),
        {"_name": "row width mismatch", "train": [[1, "a"]], "test": [], "kinds": ["numeric"]},
        {"_name": "string in a numeric column", "train": [["no"]], "test": [], "kinds": ["numeric"]},
    ]


def build() -> Problem:
    return Problem(
        slug="categorical_preprocessing",
        title="Missing values and unseen categories",
        difficulty="medium",
        summary="Impute and one-hot encode mixed columns from the training rows only, with an unknown bucket.",
        params=["train", "test", "kinds"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["preprocessing", "imputation", "categorical"],
    )
