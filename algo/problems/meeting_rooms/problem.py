import random

from gymkit.spec import Problem


def reference(meetings):
    starts = sorted(s for s, _ in meetings)
    ends = sorted(e for _, e in meetings)
    rooms = best = 0
    j = 0
    for s in starts:
        while ends[j] <= s:  # a room freed at s can be reused at s
            j += 1
            rooms -= 1
        rooms += 1
        best = max(best, rooms)
    return best


def _random_case(seed, n, span, max_len, name):
    rng = random.Random(seed)
    meetings = []
    for _ in range(n):
        start = rng.randint(-span // 10, span)
        meetings.append([start, start + rng.randint(1, max_len)])
    return {"_name": name, "meetings": meetings}


def functional_cases():
    return [
        {"_name": "example", "meetings": [[0, 30], [5, 10], [15, 20], [10, 15]]},
        {"_name": "empty", "meetings": []},
        {"_name": "one meeting", "meetings": [[3, 4]]},
        {"_name": "back to back share a room", "meetings": [[1, 4], [4, 6], [6, 9]]},
        {"_name": "identical meetings", "meetings": [[2, 5], [2, 5], [2, 5]]},
        {"_name": "nested", "meetings": [[0, 100], [10, 20], [12, 18], [30, 40]]},
        {"_name": "negative times", "meetings": [[-10, -5], [-6, 0], [-1, 3], [-7, -6]]},
        {"_name": "end equals another start, unsorted", "meetings": [[5, 8], [1, 5], [5, 6], [0, 2]]},
        _random_case(1, 60, 200, 30, "random n=60"),
        _random_case(2, 3000, 100000, 400, "random n=3000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 200_000, 10_000_000, 2_000, "perf")


def build() -> Problem:
    return Problem(
        slug="meeting_rooms",
        title="Meeting rooms",
        difficulty="medium",
        summary="The fewest rooms that hold every meeting, by sweeping start and end events.",
        params=["meetings"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        tags=["sorting", "sweep line", "heap"],
    )
