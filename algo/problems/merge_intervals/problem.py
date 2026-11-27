import random

from gymkit.spec import Problem


def reference(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _random_case(seed, n, span, max_len, name):
    rng = random.Random(seed)
    intervals = []
    for _ in range(n):
        start = rng.randint(0, span)
        intervals.append([start, start + rng.randint(0, max_len)])
    return {"_name": name, "intervals": intervals}


def functional_cases():
    return [
        {"_name": "example", "intervals": [[8, 10], [1, 3], [2, 6], [15, 18]]},
        {"_name": "empty", "intervals": []},
        {"_name": "single", "intervals": [[4, 7]]},
        {"_name": "touching ends merge", "intervals": [[1, 4], [4, 5], [6, 6]]},
        {"_name": "nested", "intervals": [[1, 10], [2, 3], [4, 5], [9, 10]]},
        {"_name": "duplicates and points", "intervals": [[3, 3], [3, 3], [1, 2], [1, 2], [0, 0]]},
        {"_name": "negative coordinates", "intervals": [[-5, -2], [-3, 0], [2, 4], [-10, -8]]},
        _random_case(1, 50, 200, 10, "random n=50"),
        _random_case(2, 2000, 100000, 60, "random n=2000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200_000, 10_000_000, 80, "perf")


def build() -> Problem:
    return Problem(
        slug="merge_intervals",
        title="Merge intervals",
        difficulty="easy",
        summary="Merge overlapping or touching closed intervals into a sorted, disjoint list.",
        params=["intervals"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        tags=["sorting", "intervals"],
    )
