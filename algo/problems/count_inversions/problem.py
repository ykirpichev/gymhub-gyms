import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 1000, "cpp": 40}


def reference(nums):
    # Merge sort that counts, for each element taken from the right half, how many
    # left-half elements are still waiting (all of them are larger).
    a = list(nums)
    n = len(a)
    buf = [0] * n
    count = 0
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i, j, t = lo, mid, lo
            while i < mid and j < hi:
                if a[j] < a[i]:
                    buf[t] = a[j]
                    count += mid - i
                    j += 1
                else:
                    buf[t] = a[i]
                    i += 1
                t += 1
            buf[t : t + mid - i] = a[i:mid]
            t += mid - i
            buf[t : t + hi - j] = a[j:hi]
        a, buf = buf, a
        width *= 2
    return count


def _random_case(seed, n, low, high, name):
    rng = random.Random(seed)
    return {"_name": name, "nums": [rng.randint(low, high) for _ in range(n)]}


def functional_cases():
    return [
        {"_name": "example", "nums": [3, 1, 2, 5, 4]},
        {"_name": "empty", "nums": []},
        {"_name": "single", "nums": [7]},
        {"_name": "sorted", "nums": list(range(20))},
        {"_name": "reversed", "nums": list(range(30, 0, -1))},
        {"_name": "all equal", "nums": [4] * 12},
        {"_name": "duplicates", "nums": [2, 2, 1, 1, 3, 2, 1]},
        _random_case(1, 200, -100, 100, "random n=200"),
        _random_case(2, 3000, 0, 50, "random n=3000, many duplicates"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200_000, -10**9, 10**9, "perf")


def build() -> Problem:
    return Problem(
        slug="count_inversions",
        title="Count inversions",
        difficulty="medium",
        summary="Count pairs i < j with a[i] > a[j] in O(n log n) with merge sort or a Fenwick tree.",
        params=["nums"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["divide and conquer", "merge sort", "fenwick tree"],
    )

