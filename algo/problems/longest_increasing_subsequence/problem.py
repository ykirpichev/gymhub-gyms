import bisect
import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 300}


def reference(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def _random_case(seed, n, span, name):
    rng = random.Random(seed)
    return {"_name": name, "nums": [rng.randint(-span, span) for _ in range(n)]}


def functional_cases():
    return [
        {"_name": "example", "nums": [10, 9, 2, 5, 3, 7, 101, 18]},
        {"_name": "empty", "nums": []},
        {"_name": "one value", "nums": [5]},
        {"_name": "all equal: strictly increasing means 1", "nums": [7, 7, 7, 7]},
        {"_name": "increasing", "nums": [1, 2, 3, 4, 5, 6]},
        {"_name": "decreasing", "nums": [6, 5, 4, 3, 2, 1]},
        {"_name": "duplicates inside a run", "nums": [1, 3, 3, 3, 4, 4, 5]},
        {"_name": "negatives", "nums": [-3, -10, -2, -2, 0, -1, 4]},
        {"_name": "a late low start wins", "nums": [50, 60, 1, 2, 3, 70, 4]},
        _random_case(1, 60, 30, "random n=60"),
        _random_case(2, 3000, 1_000_000, "random n=3000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200_000, 1_000_000_000, "perf")


def build() -> Problem:
    return Problem(
        slug="longest_increasing_subsequence",
        title="Longest increasing subsequence",
        difficulty="medium",
        summary="The length of the longest strictly increasing subsequence in O(n log n), with patience sorting.",
        params=["nums"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["dynamic programming", "binary search"],
    )
