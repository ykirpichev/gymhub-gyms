"""Grader for reservoir_sampling. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Randomness is never compared draw for draw. Every call is checked for validity (size, items come
from the stream, no position used twice) and reproducibility (same seed, same sample). For short
streams of distinct items the learner's function also runs over TRIALS fixed seeds, and each item's
inclusion count must be within Z_LIMIT standard deviations of TRIALS * k / n. For {"range": n}
streams the peak memory of one call is measured, so collecting the whole stream fails.
"""

import math
import tracemalloc
from collections import Counter

from gymkit.spec import Problem

TRIALS = 8000
Z_LIMIT = 5.0
STATS_MAX_N = 30
MEMORY_MIN_N = 10_000
MEMORY_LIMIT_BYTES = 256 * 1024


def _items(stream):
    return range(stream["range"]) if isinstance(stream, dict) else stream


def _generator(stream):
    """A one-pass iterator with no len(), as a real stream would be."""
    for item in _items(stream):
        yield item


def reference(stream, k, seed):
    if k < 0:
        return "ValueError"
    n = len(_items(stream))
    return {"size": min(k, n)}


def _key(item):
    return repr(item)


def _run(module, stream, k, seed):
    try:
        return module.solve(_generator(stream), k, seed)
    except TypeError as exc:
        if "len()" in str(exc) or "subscriptable" in str(exc):
            raise AssertionError(f"the stream is an iterator: it has no len() and no indexing ({exc})") from None
        raise


def _check_valid(sample, stream, k, seed):
    if not isinstance(sample, list):
        raise AssertionError(f"seed {seed}: return a list, got {type(sample).__name__}")
    items = list(_items(stream))
    want = min(k, len(items))
    if len(sample) != want:
        raise AssertionError(f"seed {seed}: returned {len(sample)} items, expected min(k, n) = {want}")
    available = Counter(_key(x) for x in items)
    used = Counter(_key(x) for x in sample)
    for key, count in used.items():
        if key not in available:
            raise AssertionError(f"seed {seed}: {key} is not in the stream")
        if count > available[key]:
            raise AssertionError(f"seed {seed}: {key} appears {count} times in the sample but only "
                                 f"{available[key]} times in the stream: a stream position was used twice")


def call(module, args):
    stream, k, seed = args
    try:
        sample = _run(module, stream, k, seed)
    except ValueError:
        return "ValueError"
    if k < 0:
        return sample  # compare reports the missing ValueError
    _check_valid(sample, stream, k, seed)
    again = _run(module, stream, k, seed)
    if sorted(map(_key, again)) != sorted(map(_key, sample)):
        raise AssertionError(f"seed {seed}: two calls with the same seed returned different samples. "
                             "Draw every random number from random.Random(seed)")

    items = list(_items(stream))
    n = len(items)
    if isinstance(stream, list) and n <= STATS_MAX_N and 0 < k < n and len(set(map(_key, items))) == n:
        counts = Counter()
        for trial in range(TRIALS):
            trial_seed = 10_000 + trial
            got = _run(module, stream, k, trial_seed)
            _check_valid(got, stream, k, trial_seed)
            counts.update(map(_key, got))
        p = k / n
        mean, sd = TRIALS * p, math.sqrt(TRIALS * p * (1 - p))
        for position, item in enumerate(items):
            z = (counts[_key(item)] - mean) / sd
            if abs(z) > Z_LIMIT:
                table = ", ".join(f"{counts[_key(x)] / TRIALS:.3f}" for x in items)
                raise AssertionError(
                    f"not uniform: over {TRIALS} seeds, stream position {position} ({item!r}) was kept "
                    f"{counts[_key(item)] / TRIALS:.3f} of the time, expected k/n = {p:.3f} ({z:+.1f} sd)\n"
                    f"inclusion rate by position: {table}")

    if isinstance(stream, dict) and n >= MEMORY_MIN_N:
        tracemalloc.start()
        try:
            _run(module, stream, k, seed)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        if peak > MEMORY_LIMIT_BYTES + 200 * k:
            raise AssertionError(f"peak memory {peak / 1024:.0f} KiB for a stream of {n} items with k = {k}: "
                                 "keep only the k-item reservoir, not the whole stream")
    return {"size": len(sample)}


def compare(got, want, case):
    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError("expected ValueError for k < 0" if want == "ValueError"
                                 else "raised ValueError, but the input is valid")
        return
    if got != want:
        raise AssertionError(f"got {got!r}, expected {want!r}")


def functional_cases():
    return [
        {"_name": "example", "stream": ["a", "b", "c", "d", "e"], "k": 2, "seed": 7},
        {"_name": "k = 1 of 4", "stream": [10, 20, 30, 40], "k": 1, "seed": 0},
        {"_name": "k = 3 of 10", "stream": list(range(10)), "k": 3, "seed": 1},
        {"_name": "k = 5 of 20 strings", "stream": [f"x{i}" for i in range(20)], "k": 5, "seed": 2},
        {"_name": "k = n - 1", "stream": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5], "k": 5, "seed": 3},
        {"_name": "stream shorter than k", "stream": [4, 5, 6], "k": 10, "seed": 4},
        {"_name": "k equals stream length", "stream": [4, 5, 6], "k": 3, "seed": 5},
        {"_name": "empty stream", "stream": [], "k": 3, "seed": 6},
        {"_name": "k = 0", "stream": [1, 2, 3], "k": 0, "seed": 7},
        {"_name": "repeated values", "stream": [1, 1, 1, 2, 2, 3], "k": 4, "seed": 8},
        {"_name": "long stream, O(k) memory", "stream": {"range": 100_000}, "k": 10, "seed": 9},
        {"_name": "negative k", "stream": [1, 2, 3], "k": -1, "seed": 10},
    ]


def build() -> Problem:
    return Problem(
        slug="reservoir_sampling",
        title="Reservoir sampling",
        difficulty="medium",
        summary="Keep a uniform random sample of k items from a stream of unknown length in O(k) memory.",
        params=["stream", "k", "seed"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["streaming", "probability", "sampling"],
    )
