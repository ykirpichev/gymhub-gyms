"""Grader for sampling_strategies. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Samples are never compared to one random stream. Every sample is checked for validity (length,
range, replacement rules, zero weights, stratified counts). The same seed must repeat. Over TRIALS
fixed seeds, uniform and weighted draws are checked against their probabilities, and a stratified
sample must shuffle inside each class.
"""

import copy
import math
from collections import Counter, defaultdict

import numpy as np

from gymkit.spec import Problem

TRIALS = 2000
Z_LIMIT = 6.0
MODES = ("uniform_with_replacement", "uniform_without_replacement", "weighted", "stratified")


def _invalid(n, k, mode, weights, labels):
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        return "n must be a non-negative integer"
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        return "k must be a non-negative integer"
    if mode not in MODES:
        return f"unknown mode {mode!r}"
    if k > 0 and n == 0:
        return "cannot draw from an empty population"
    if mode in ("uniform_without_replacement", "stratified") and k > n:
        return "k exceeds n for a sample without replacement"
    if mode == "weighted":
        if not isinstance(weights, (list, tuple)) or len(weights) != n:
            return "weights must have one number per index"
        if any(isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0 for w in weights):
            return "weights must be non-negative"
        if k > 0 and math.fsum(float(w) for w in weights) <= 0:
            return "weights must sum to a positive number"
    if mode == "stratified" and (not isinstance(labels, (list, tuple)) or len(labels) != n):
        return "labels must have one label per index"
    return None


def reference(n, k, mode, seed, weights, labels):
    if _invalid(n, k, mode, weights, labels):
        return "ValueError"
    return {"length": k}


def _as_indices(sample):
    if isinstance(sample, np.ndarray):
        sample = sample.tolist()
    if not isinstance(sample, (list, tuple)):
        raise AssertionError(f"return a list of indices, got {type(sample).__name__}")
    out = []
    for item in sample:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise AssertionError(f"sample contains {item!r}, expected an integer index")
        out.append(int(item))
    return out


def _check_sample(sample, n, k, mode, seed, weights, labels):
    if len(sample) != k:
        raise AssertionError(f"seed {seed}: returned {len(sample)} indices, expected {k}")
    if any(index < 0 or index >= n for index in sample):
        raise AssertionError(f"seed {seed}: an index is outside 0..{n - 1}: {sample}")
    distinct = len(set(sample)) == len(sample)
    if mode in ("uniform_without_replacement", "stratified") and not distinct:
        raise AssertionError(f"seed {seed}: {mode} repeated an index in {sample}")
    if mode == "weighted":
        for index in sample:
            if weights[index] <= 0:
                raise AssertionError(f"seed {seed}: index {index} has weight {weights[index]} and must never be drawn")
        positive = sum(1 for weight in weights if weight > 0)
        if k > positive and distinct:
            raise AssertionError(
                f"seed {seed}: weighted sampling is with replacement, but {k} draws from "
                f"{positive} positive weights were all distinct"
            )
    if mode == "stratified" and k > 0:
        seen = Counter(labels[index] for index in sample)
        grouped = Counter(labels)
        if sum(seen.values()) != k:
            raise AssertionError(f"seed {seed}: stratified counts do not sum to k")
        for label, n_c in grouped.items():
            count = seen[label]
            expected = k * n_c / n
            if count > n_c:
                raise AssertionError(f"seed {seed}: class {label!r} was drawn {count} times from only {n_c} rows")
            if abs(count - expected) > 1 + 1e-9:
                raise AssertionError(
                    f"seed {seed}: class {label!r} appears {count} times, expected within ±1 of {expected:.3f}"
                )
            if k * n_c >= n and count < 1:
                raise AssertionError(
                    f"seed {seed}: class {label!r} has {n_c} rows and k * n_c >= n, so the sample needs at least one"
                )


def _binom(count, trials, p, what):
    mean = trials * p
    var = trials * p * (1.0 - p)
    if var < 1e-9:
        if count != int(round(mean)):
            raise AssertionError(f"{what}: count {count}, expected {mean:g} over {trials} draws")
        return
    z = (count - mean) / math.sqrt(var)
    if abs(z) > Z_LIMIT:
        raise AssertionError(f"{what}: rate {count / trials:.3f}, expected {p:.3f} ({z:+.1f} sd over {trials})")


def _p_duplicate(n, k):
    if k <= 1:
        return 0.0
    if k > n:
        return 1.0
    distinct = 1.0
    for i in range(k):
        distinct *= (n - i) / n
    return 1.0 - distinct


def _solve(module, n, k, mode, seed, weights, labels):
    try:
        sample = module.solve(n, k, mode, seed, copy.deepcopy(weights), copy.deepcopy(labels))
    except ValueError as exc:
        raise AssertionError(f"seed {seed}: raised ValueError on a valid sample ({exc})") from None
    parsed = _as_indices(sample)
    _check_sample(parsed, n, k, mode, seed, weights, labels)
    return parsed


def _distribution(module, n, k, mode, weights, labels):
    occurrences = Counter()
    duplicate_trials = 0
    for trial in range(TRIALS):
        sample = _solve(module, n, k, mode, 10_000 + trial, weights, labels)
        occurrences.update(sample)
        if len(sample) != len(set(sample)):
            duplicate_trials += 1
    if mode == "uniform_without_replacement":
        for index in range(n):
            _binom(occurrences[index], TRIALS, k / n, f"index {index} inclusion")
        return
    if mode == "uniform_with_replacement":
        for index in range(n):
            _binom(occurrences[index], TRIALS * k, 1 / n, f"index {index} draw rate")
        _binom(duplicate_trials, TRIALS, _p_duplicate(n, k), "samples that repeat an index")
        return
    if mode == "weighted":
        total = math.fsum(float(weight) for weight in weights)
        for index, weight in enumerate(weights):
            _binom(occurrences[index], TRIALS * k, float(weight) / total, f"index {index} draw rate")
        return
    members = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    for label, group in members.items():
        if len(group) < 2:
            continue
        counts = [occurrences[index] for index in group]
        mean = sum(counts) / len(counts)
        p = mean / TRIALS
        var = TRIALS * p * (1.0 - p)
        if var < 1.0:
            if any(count != counts[0] for count in counts):
                raise AssertionError(
                    f"class {label!r}: members are not exchangeable, inclusion counts {counts}. Shuffle inside the class."
                )
            continue
        sd = math.sqrt(var)
        for index, count in zip(group, counts):
            z = (count - mean) / sd
            if abs(z) > Z_LIMIT:
                raise AssertionError(
                    f"class {label!r}: index {index} was included {count / TRIALS:.3f} of the time, "
                    f"class average {p:.3f} ({z:+.1f} sd). Shuffle inside the class."
                )


def call(module, args):
    n, k, mode, seed, weights, labels = args
    reason = _invalid(n, k, mode, weights, labels)
    try:
        sample = module.solve(n, k, mode, seed, copy.deepcopy(weights), copy.deepcopy(labels))
    except ValueError:
        return "ValueError"
    if reason:
        raise AssertionError(f"expected ValueError: {reason}")
    parsed = _as_indices(sample)
    _check_sample(parsed, n, k, mode, seed, weights, labels)
    again = _solve(module, n, k, mode, seed, weights, labels)
    if again != parsed:
        raise AssertionError("same seed returned a different sample. Draw every random number from random.Random(seed)")
    if 0 < k <= 40 and n <= 40:
        _distribution(module, n, k, mode, weights, labels)
    return {"length": k}


def compare(got, want, case):
    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError" if want == "ValueError" else "raised ValueError, but the sample request is valid"
            )
        return
    if got != want:
        raise AssertionError(f"got {got!r}, expected {want!r}")


def _random_weighted():
    rng = np.random.default_rng(19)
    n = int(rng.integers(8, 15))
    weights = [int(value) for value in rng.integers(1, 8, size=n)]
    weights[int(rng.integers(0, n))] = 0
    return {
        "_name": f"random weighted n={n}",
        "n": n,
        "k": 1,
        "mode": "weighted",
        "seed": 19,
        "weights": weights,
        "labels": None,
    }


def _random_stratified():
    rng = np.random.default_rng(23)
    labels = np.array([0] * 10 + [1] * 6)
    rng.shuffle(labels)
    return {
        "_name": "random stratified positions",
        "n": 16,
        "k": 8,
        "mode": "stratified",
        "seed": 23,
        "weights": None,
        "labels": labels.astype(int).tolist(),
    }


def functional_cases():
    return [
        {"_name": "example", "n": 6, "k": 2, "mode": "uniform_without_replacement",
         "seed": 7, "weights": None, "labels": None},
        {"_name": "with replacement longer than n", "n": 5, "k": 8, "mode": "uniform_with_replacement",
         "seed": 1, "weights": None, "labels": None},
        {"_name": "without replacement takes everyone", "n": 4, "k": 4, "mode": "uniform_without_replacement",
         "seed": 2, "weights": None, "labels": None},
        {"_name": "weighted zeros and a 1:3 ratio", "n": 4, "k": 6, "mode": "weighted",
         "seed": 3, "weights": [1, 0, 3, 0], "labels": None},
        {"_name": "negative weight", "n": 3, "k": 2, "mode": "weighted",
         "seed": 4, "weights": [1, -2, 3], "labels": None},
        {"_name": "weights sum to zero", "n": 3, "k": 1, "mode": "weighted",
         "seed": 5, "weights": [0, 0, 0], "labels": None},
        {"_name": "stratified rare class", "n": 10, "k": 5, "mode": "stratified",
         "seed": 6, "weights": None, "labels": [0] * 8 + [1] * 2},
        {"_name": "stratified balance", "n": 12, "k": 6, "mode": "stratified",
         "seed": 7, "weights": None, "labels": [0] * 6 + [1] * 6},
        {"_name": "k = 0", "n": 5, "k": 0, "mode": "uniform_with_replacement",
         "seed": 8, "weights": None, "labels": None},
        {"_name": "without replacement k > n", "n": 3, "k": 4, "mode": "uniform_without_replacement",
         "seed": 9, "weights": None, "labels": None},
        {"_name": "weights length mismatch", "n": 3, "k": 1, "mode": "weighted",
         "seed": 10, "weights": [1, 2], "labels": None},
        {"_name": "labels length mismatch", "n": 4, "k": 2, "mode": "stratified",
         "seed": 11, "weights": None, "labels": [0, 1]},
        {"_name": "unknown mode", "n": 3, "k": 1, "mode": "bootstrap",
         "seed": 12, "weights": None, "labels": None},
        _random_weighted(),
        _random_stratified(),
    ]


def build() -> Problem:
    return Problem(
        slug="sampling_strategies",
        title="Uniform, weighted, and stratified sampling",
        difficulty="medium",
        summary="Draw uniform samples with or without replacement, weighted samples, or a stratified sample with a per-class minimum.",
        params=["n", "k", "mode", "seed", "weights", "labels"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["sampling", "weights", "stratification"],
    )
