import heapq
import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 800}


def reference(n, prerequisites):
    after = [[] for _ in range(n)]
    waiting = [0] * n
    for a, b in prerequisites:
        after[b].append(a)
        waiting[a] += 1
    ready = [c for c in range(n) if waiting[c] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        c = heapq.heappop(ready)
        order.append(c)
        for d in after[c]:
            waiting[d] -= 1
            if waiting[d] == 0:
                heapq.heappush(ready, d)
    return order if len(order) == n else []


def _dag_case(seed, n, m, name):
    rng = random.Random(seed)
    rank = list(range(n))
    rng.shuffle(rank)  # a hidden order; edges only go forward in it, so there is no cycle
    pairs = []
    for _ in range(m):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        lo, hi = min(i, j), max(i, j)
        pairs.append([rank[hi], rank[lo]])  # take rank[lo] before rank[hi]
    return {"_name": name, "n": n, "prerequisites": pairs}


def functional_cases():
    cyclic = _dag_case(3, 40, 80, "random with a cycle")
    cyclic["prerequisites"].extend([[5, 6], [6, 7], [7, 5]])
    return [
        {"_name": "example", "n": 4, "prerequisites": [[1, 0], [2, 0], [3, 1], [3, 2]]},
        {"_name": "two-course cycle", "n": 2, "prerequisites": [[0, 1], [1, 0]]},
        {"_name": "one course", "n": 1, "prerequisites": []},
        {"_name": "no prerequisites: 0..n-1", "n": 5, "prerequisites": []},
        {"_name": "chain backwards", "n": 5, "prerequisites": [[0, 1], [1, 2], [2, 3], [3, 4]]},
        {"_name": "smallest ready first", "n": 6, "prerequisites": [[0, 5], [1, 4], [2, 3]]},
        {"_name": "repeated pairs", "n": 3, "prerequisites": [[2, 0], [2, 0], [1, 2]]},
        {"_name": "self-loop", "n": 3, "prerequisites": [[1, 1]]},
        cyclic,
        _dag_case(1, 50, 120, "random DAG n=50"),
        _dag_case(2, 3000, 9000, "random DAG n=3000"),
    ]


def performance_case(seed: int = 0):
    return _dag_case(seed, 100_000, 300_000, "perf")


def build() -> Problem:
    return Problem(
        slug="course_schedule",
        title="Course schedule",
        difficulty="medium",
        summary="Topological order of tasks with prerequisites, smallest label first, or an empty list on a cycle.",
        params=["n", "prerequisites"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["graphs", "topological sort", "heap"],
    )
