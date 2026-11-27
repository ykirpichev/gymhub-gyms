import random
from collections import deque

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 100, "cpp": 6}


def reference(nums, k):
    window = deque()  # indices, values decreasing from front to back
    out = []
    for i, value in enumerate(nums):
        while window and nums[window[-1]] <= value:
            window.pop()
        window.append(i)
        if window[0] <= i - k:
            window.popleft()
        if i >= k - 1:
            out.append(nums[window[0]])
    return out


def _random_case(seed, n, k, low, high, name):
    rng = random.Random(seed)
    return {"_name": name, "nums": [rng.randint(low, high) for _ in range(n)], "k": k}


def functional_cases():
    return [
        {"_name": "example", "nums": [1, 3, -1, -3, 5, 3, 6, 7], "k": 3},
        {"_name": "empty", "nums": [], "k": 1},
        {"_name": "k = 1", "nums": [4, -2, 7, 0], "k": 1},
        {"_name": "k = len", "nums": [2, 9, 4, 9, 1], "k": 5},
        {"_name": "decreasing", "nums": [9, 8, 7, 6, 5, 4, 3], "k": 3},
        {"_name": "increasing", "nums": [1, 2, 3, 4, 5, 6, 7], "k": 4},
        {"_name": "all equal", "nums": [5] * 10, "k": 4},
        _random_case(1, 100, 7, -50, 50, "random n=100 k=7"),
        _random_case(2, 5000, 300, 0, 10, "random n=5000 k=300, many duplicates"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200_000, 5_000, -10**9, 10**9, "perf")


def build() -> Problem:
    return Problem(
        slug="sliding_window_max",
        title="Sliding window maximum",
        difficulty="medium",
        summary="Maximum of every length-k window in one pass with a monotonic deque.",
        params=["nums", "k"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["deque", "monotonic queue", "sliding window"],
    )

