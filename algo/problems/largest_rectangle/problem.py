import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 200}


def reference(heights):
    stack = []  # indices with increasing heights
    best = 0
    for i, h in enumerate(list(heights) + [0]):
        while stack and heights[stack[-1]] >= h:
            top = heights[stack.pop()]
            left = stack[-1] if stack else -1
            best = max(best, top * (i - left - 1))
        stack.append(i)
    return best


def _random_case(seed, n, high, name):
    rng = random.Random(seed)
    return {"_name": name, "heights": [rng.randint(0, high) for _ in range(n)]}


def functional_cases():
    return [
        {"_name": "example", "heights": [2, 1, 5, 6, 2, 3]},
        {"_name": "empty", "heights": []},
        {"_name": "one bar", "heights": [7]},
        {"_name": "all equal", "heights": [3, 3, 3, 3]},
        {"_name": "increasing", "heights": [1, 2, 3, 4, 5]},
        {"_name": "decreasing", "heights": [5, 4, 3, 2, 1]},
        {"_name": "zeros split the chart", "heights": [4, 4, 0, 3, 3, 3, 0, 1]},
        {"_name": "valley", "heights": [6, 2, 1, 2, 6]},
        _random_case(1, 40, 10, "random n=40"),
        _random_case(2, 3000, 100000, "random n=3000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200_000, 1_000_000_000, "perf")


def build() -> Problem:
    return Problem(
        slug="largest_rectangle",
        title="Largest rectangle in a histogram",
        difficulty="hard",
        summary="The largest axis-aligned rectangle under a bar chart, in one pass with a monotonic stack.",
        params=["heights"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["stack", "monotonic stack", "arrays"],
    )
