"""Grader for dataset_split. Private: learners see PROBLEM.md, HINTS.md and the starter only.

The split is not compared to one permutation. Each call is checked for coverage, the floor/leftover
sizes, and (when stratified) the per-class rules. The same seed must repeat. Over TRIALS fixed
seeds a plain split's inclusion rate is tested, and a stratified split must shuffle inside each class.
"""

import copy
import math
from collections import Counter

import numpy as np

from gymkit.spec import Problem

TRIALS = 2000
Z_LIMIT = 6.0
STATS_MAX_N = 48


def _sizes(n, fractions):
    sizes = [math.floor(n * float(f) + 1e-12) for f in fractions]
    leftover = n - sum(sizes)
    if leftover < 0:
        raise ValueError("fractions are too large for n")
    for i in range(leftover):
        sizes[i % 3] += 1
    return sizes


def _invalid(n, fractions, labels):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return "n must be a non-negative integer"
    if not isinstance(fractions, (list, tuple)) or len(fractions) != 3:
        return "fractions must be three non-negative numbers"
    if any(isinstance(f, bool) or not isinstance(f, (int, float)) or f < 0 for f in fractions):
        return "fractions must be three non-negative numbers"
    try:
        _sizes(n, fractions)
    except ValueError as exc:
        return str(exc)
    if labels is not None and len(labels) != n:
        return "labels must have length n"
    if labels is not None and n and not _feasible(Counter(labels).values(), _sizes(n, fractions), n):
        return "these class sizes cannot meet the split rules"
    return None


def _class_options(count, sizes, n):
    """Per-split counts (train, validation, test) that satisfy the rules _check enforces for one class."""
    if count == 1:
        candidates = [(1, 0, 0)]
    elif count == 2:
        candidates = [(1, 1, 0)]
    else:
        candidates = [(a, b, count - a - b) for a in range(1, count) for b in range(1, count - a)]
    return [c for c in candidates if all(abs(c[s] - count * sizes[s] / n) <= 1 + 1e-9 for s in range(3))]


def _feasible(class_counts, sizes, n):
    """Whether some per-class allocation meets every rule and fills the splits exactly (a small DP over classes)."""
    reachable = {(0, 0)}
    for count in class_counts:
        options = _class_options(count, sizes, n)
        reachable = {(t + a, v + b) for t, v in reachable for a, b, _c in options
                     if t + a <= sizes[0] and v + b <= sizes[1]}
        if not reachable:
            return False
    return (sizes[0], sizes[1]) in reachable


def reference(n, fractions, labels, seed):
    if _invalid(n, fractions, labels):
        return "ValueError"
    sizes = _sizes(n, fractions)
    return {"train": sizes[0], "validation": sizes[1], "test": sizes[2]}


def _indices(value, name):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise AssertionError(f"{name} must be a list of indices, got {type(value).__name__}")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise AssertionError(f"{name} contains {item!r}, expected an integer index")
        out.append(int(item))
    return out


def _partition(got, n):
    if not isinstance(got, dict) or set(got) != {"train", "validation", "test"}:
        raise AssertionError("return a dict with keys train, validation and test")
    parts = [_indices(got[name], name) for name in ("train", "validation", "test")]
    flat = sorted(index for part in parts for index in part)
    if flat != list(range(n)):
        raise AssertionError(
            f"the splits must be disjoint and cover 0..{n - 1}; got {len(flat)} indices "
            f"with {len(flat) - len(set(flat))} duplicates"
        )
    return parts


def _check(got, n, fractions, labels):
    sizes = _sizes(n, fractions)
    parts = _partition(got, n)
    got_sizes = [len(part) for part in parts]
    if got_sizes != sizes:
        raise AssertionError(
            f"split lengths {got_sizes}, expected {sizes} "
            "(floor(n * fraction + 1e-12), leftover to train, then validation, then test)"
        )
    if labels is None:
        return parts
    grouped = Counter(labels)
    for label, count in sorted(grouped.items(), key=lambda item: item[0]):
        counts = [sum(1 for index in part if labels[index] == label) for part in parts]
        expected = [count * sizes[s] / n for s in range(3)]
        if count == 1 and counts != [1, 0, 0]:
            raise AssertionError(f"label {label!r} has 1 row and must go to train, got counts {counts}")
        if count == 2 and counts != [1, 1, 0]:
            raise AssertionError(
                f"label {label!r} has 2 rows and must put one in train and one in validation, got counts {counts}"
            )
        if count >= 3 and any(value < 1 for value in counts):
            raise AssertionError(
                f"label {label!r} has {count} rows, so every split needs at least one; got counts {counts}"
            )
        if any(abs(counts[s] - expected[s]) > 1 + 1e-9 for s in range(3)):
            shown = [round(value, 3) for value in expected]
            raise AssertionError(
                f"label {label!r} has counts {counts}, expected within ±1 of {shown} "
                f"(class size {count}, split sizes {sizes})"
            )
    return parts


def _binom(count, trials, p, what):
    mean = trials * p
    var = trials * p * (1.0 - p)
    if var < 1e-9:
        if count != int(round(mean)):
            raise AssertionError(f"{what}: count {count}, expected {mean:g} over {trials} seeds")
        return
    z = (count - mean) / math.sqrt(var)
    if abs(z) > Z_LIMIT:
        raise AssertionError(
            f"{what}: rate {count / trials:.3f}, expected {p:.3f} ({z:+.1f} sd over {trials} seeds)"
        )


def _exchange(members, included, trials, label, split_name):
    if len(members) < 2:
        return
    counts = [included[index] for index in members]
    mean = sum(counts) / len(counts)
    p = mean / trials
    var = trials * p * (1.0 - p)
    if var < 1.0:
        if any(count != counts[0] for count in counts):
            raise AssertionError(
                f"label {label!r} in {split_name}: class members are not exchangeable, counts {counts}. "
                "Shuffle inside the class."
            )
        return
    sd = math.sqrt(var)
    for index, count in zip(members, counts):
        z = (count - mean) / sd
        if abs(z) > Z_LIMIT:
            raise AssertionError(
                f"label {label!r} in {split_name}: index {index} was included {count / trials:.3f} of the time, "
                f"class average {p:.3f} ({z:+.1f} sd). Shuffle inside the class."
            )


def _run(module, n, fractions, labels, seed):
    try:
        got = module.solve(n, copy.deepcopy(fractions), copy.deepcopy(labels), seed)
    except ValueError as exc:
        raise AssertionError(f"seed {seed}: raised ValueError on a valid split ({exc})") from None
    return _check(got, n, fractions, labels)


def call(module, args):
    n, fractions, labels, seed = args
    reason = _invalid(n, fractions, labels)
    try:
        got = module.solve(n, copy.deepcopy(fractions), copy.deepcopy(labels), seed)
    except ValueError:
        return "ValueError"
    if reason:
        raise AssertionError(f"expected ValueError: {reason}")
    first = _check(got, n, fractions, labels)
    again = _run(module, n, fractions, labels, seed)
    if again != first and [sorted(part) for part in again] != [sorted(part) for part in first]:
        raise AssertionError("same seed returned a different split. Shuffle with random.Random(seed)")
    if 0 < n <= STATS_MAX_N:
        _distribution(module, n, fractions, labels)
    return {"train": len(first[0]), "validation": len(first[1]), "test": len(first[2])}


def _distribution(module, n, fractions, labels):
    sizes = _sizes(n, fractions)
    included = [Counter() for _ in range(3)]
    names = ("train", "validation", "test")
    for trial in range(TRIALS):
        parts = _run(module, n, fractions, labels, 10_000 + trial)
        for split, part in enumerate(parts):
            included[split].update(part)
    if labels is None:
        for split, size in enumerate(sizes):
            for index in range(n):
                _binom(included[split][index], TRIALS, size / n, f"index {index} in {names[split]}")
        return
    members = {}
    for index, label in enumerate(labels):
        members.setdefault(label, []).append(index)
    for label, group in members.items():
        for split, name in enumerate(names):
            _exchange(group, included[split], TRIALS, label, name)


def compare(got, want, case):
    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError for an impossible size request" if want == "ValueError"
                else "raised ValueError, but the split request is valid"
            )
        return
    if got != want:
        raise AssertionError(f"split sizes {got}, expected {want}")


def _plain_random():
    rng = np.random.default_rng(4)
    n = int(rng.integers(28, 46))
    return {
        "_name": f"random unstratified n={n}",
        "n": n,
        "fractions": [0.5, 0.25, 0.25],
        "labels": None,
        "seed": 4,
    }


def _strat_random():
    rng = np.random.default_rng(5)
    per = int(rng.choice([8, 12]))
    labels = np.array([0] * per + [1] * per + [2] * per)
    rng.shuffle(labels)
    return {
        "_name": f"random stratified n={3 * per}",
        "n": 3 * per,
        "fractions": [0.5, 0.25, 0.25],
        "labels": labels.astype(int).tolist(),
        "seed": 5,
    }


def functional_cases():
    return [
        {"_name": "example", "n": 9, "fractions": [1 / 3, 1 / 3, 1 / 3],
         "labels": [0, 0, 0, 1, 1, 1, 2, 2, 2], "seed": 1},
        {"_name": "plain 5, 3, 2", "n": 10, "fractions": [0.5, 0.3, 0.2], "labels": None, "seed": 0},
        {"_name": "leftover to train then validation", "n": 10, "fractions": [0.33, 0.33, 0.34],
         "labels": None, "seed": 2},
        {"_name": "fractions sum to 0", "n": 5, "fractions": [0, 0, 0], "labels": None, "seed": 3},
        {"_name": "everything in train", "n": 4, "fractions": [1, 0, 0], "labels": None, "seed": 4},
        {"_name": "empty", "n": 0, "fractions": [0.5, 0.25, 0.25], "labels": None, "seed": 5},
        {"_name": "single index", "n": 1, "fractions": [0.5, 0.25, 0.25], "labels": None, "seed": 6},
        {"_name": "small classes", "n": 8, "fractions": [0.5, 0.25, 0.25],
         "labels": [0, 1, 1, 2, 2, 2, 2, 2], "seed": 7},
        {"_name": "string labels", "n": 12, "fractions": [0.5, 0.25, 0.25],
         "labels": ["cat"] * 6 + ["dog"] * 6, "seed": 8},
        {"_name": "rare class in every split", "n": 18, "fractions": [0.5, 0.25, 0.25],
         "labels": [0] * 15 + [1] * 3, "seed": 9},
        _plain_random(),
        _strat_random(),
        {"_name": "labels length mismatch", "n": 4, "fractions": [0.5, 0.25, 0.25], "labels": [0, 1], "seed": 0},
        {"_name": "negative fraction", "n": 4, "fractions": [0.8, -0.1, 0.3], "labels": None, "seed": 0},
        {"_name": "fractions too large", "n": 5, "fractions": [0.9, 0.9, 0.9], "labels": None, "seed": 0},
        {"_name": "classes cannot meet the split rules", "n": 4, "fractions": [1, 0, 0], "labels": [0, 0, 0, 1], "seed": 0},
    ]


def build() -> Problem:
    return Problem(
        slug="dataset_split",
        title="Train/validation/test split without leakage",
        difficulty="easy",
        summary="Reproducible disjoint train, validation and test indices, optionally stratified by label.",
        params=["n", "fractions", "labels", "seed"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["data split", "stratification", "reproducibility"],
    )
