import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 1000}


def reference(weights, values, capacity):
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            if best[c - w] + v > best[c]:
                best[c] = best[c - w] + v
    return best[capacity]


def _random_case(seed, n, max_w, capacity, name):
    rng = random.Random(seed)
    weights = [rng.randint(1, max_w) for _ in range(n)]
    values = [rng.randint(0, 3 * w + 10) for w in weights]
    return {"_name": name, "weights": weights, "values": values, "capacity": capacity}


def functional_cases():
    return [
        {"_name": "example", "weights": [1, 3, 4, 5], "values": [1, 4, 5, 7], "capacity": 7},
        {"_name": "no items", "weights": [], "values": [], "capacity": 10},
        {"_name": "zero capacity", "weights": [1, 2], "values": [5, 6], "capacity": 0},
        {"_name": "item too heavy", "weights": [11], "values": [100], "capacity": 10},
        {"_name": "everything fits", "weights": [2, 3, 4], "values": [3, 4, 5], "capacity": 100},
        {"_name": "each item once", "weights": [3], "values": [10], "capacity": 9},
        {"_name": "greedy by ratio is wrong", "weights": [6, 5, 5], "values": [8, 6, 6], "capacity": 10},
        {"_name": "zero-value items", "weights": [1, 1, 1], "values": [0, 0, 0], "capacity": 2},
        _random_case(1, 12, 15, 40, "random 12 items"),
        _random_case(2, 60, 100, 1000, "random 60 items"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200, 500, 10_000, "perf")


def build() -> Problem:
    return Problem(
        slug="knapsack",
        title="0/1 knapsack",
        difficulty="medium",
        summary="The most value that fits in a weight budget, each item taken at most once: a one-row DP.",
        params=["weights", "values", "capacity"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["dynamic programming"],
    )
