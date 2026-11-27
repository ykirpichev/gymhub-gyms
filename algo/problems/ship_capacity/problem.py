import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 600}


def _days_needed(weights, capacity):
    days, load = 1, 0
    for w in weights:
        if load + w > capacity:
            days += 1
            load = 0
        load += w
    return days


def reference(weights, days):
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if _days_needed(weights, mid) <= days:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _random_case(seed, n, high, days, name):
    rng = random.Random(seed)
    return {"_name": name, "weights": [rng.randint(1, high) for _ in range(n)], "days": days}


def functional_cases():
    return [
        {"_name": "example", "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "days": 5},
        {"_name": "one package", "weights": [7], "days": 1},
        {"_name": "one day: the total", "weights": [3, 2, 2, 4, 1, 4], "days": 1},
        {"_name": "a day per package: the heaviest", "weights": [3, 2, 2, 4, 1, 4], "days": 6},
        {"_name": "three days", "weights": [3, 2, 2, 4, 1, 4], "days": 3},
        {"_name": "equal weights", "weights": [5, 5, 5, 5, 5, 5, 5], "days": 3},
        {"_name": "one heavy package", "weights": [1, 1, 1, 100, 1, 1, 1], "days": 4},
        _random_case(1, 50, 20, 7, "random n=50"),
        _random_case(2, 2000, 1000, 37, "random n=2000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 100_000, 10_000, 500, "perf")


def build() -> Problem:
    return Problem(
        slug="ship_capacity",
        title="Minimum shipping capacity",
        difficulty="medium",
        summary="Binary search on the answer: the smallest daily capacity that ships every package in order within D days.",
        params=["weights", "days"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["binary search", "greedy"],
    )
