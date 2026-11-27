import heapq
import random

from gymkit.spec import Problem


def reference(nums):
    low, high = [], []  # low is a max-heap (negated), high a min-heap
    out = []
    for x in nums:
        heapq.heappush(low, -x)
        heapq.heappush(high, -heapq.heappop(low))
        if len(high) > len(low):
            heapq.heappush(low, -heapq.heappop(high))
        out.append(float(-low[0]) if len(low) > len(high) else (-low[0] + high[0]) / 2.0)
    return out


def _random_case(seed, n, span, name):
    rng = random.Random(seed)
    return {"_name": name, "nums": [rng.randint(-span, span) for _ in range(n)]}


def functional_cases():
    return [
        {"_name": "example", "nums": [5, 15, 1, 3]},
        {"_name": "empty", "nums": []},
        {"_name": "one number", "nums": [42]},
        {"_name": "duplicates", "nums": [2, 2, 2, 1, 1, 3, 3]},
        {"_name": "negatives", "nums": [-5, -1, -10, 0, 7, -3]},
        {"_name": "increasing", "nums": [1, 2, 3, 4, 5, 6, 7, 8]},
        {"_name": "decreasing", "nums": [8, 7, 6, 5, 4, 3, 2, 1]},
        {"_name": "odd average", "nums": [1, 2]},
        _random_case(1, 50, 20, "random n=50"),
        _random_case(2, 3000, 1_000_000, "random n=3000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 100_000, 1_000_000_000, "perf")


def build() -> Problem:
    return Problem(
        slug="running_median",
        title="Running median",
        difficulty="medium",
        summary="The median after every new number in a stream, with two heaps.",
        params=["nums"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        tags=["heap", "streaming"],
    )
