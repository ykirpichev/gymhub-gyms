import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 1500}


def reference(n, edges):
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total, taken = 0, 0
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru
        size[ru] += size[rv]
        total += w
        taken += 1
        if taken == n - 1:
            break
    return total if taken == n - 1 else -1


def _connected_case(seed, n, m, low, high, name):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = [[order[i], order[rng.randrange(i)], rng.randint(low, high)] for i in range(1, n)]  # a random tree
    while len(edges) < m:
        edges.append([rng.randrange(n), rng.randrange(n), rng.randint(low, high)])
    rng.shuffle(edges)
    return {"_name": name, "n": n, "edges": edges}


def functional_cases():
    return [
        {"_name": "example", "n": 4, "edges": [[0, 1, 1], [1, 2, 4], [0, 2, 3], [2, 3, 2], [1, 3, 5]]},
        {"_name": "one node", "n": 1, "edges": []},
        {"_name": "one node with a self-loop", "n": 1, "edges": [[0, 0, -5]]},
        {"_name": "two nodes, parallel edges", "n": 2, "edges": [[0, 1, 9], [1, 0, 3], [0, 1, 7]]},
        {"_name": "disconnected", "n": 4, "edges": [[0, 1, 1], [2, 3, 1]]},
        {"_name": "isolated node", "n": 3, "edges": [[0, 1, 2], [1, 0, 1]]},
        {"_name": "self-loops are never taken", "n": 3, "edges": [[0, 0, -10], [0, 1, 4], [1, 2, 6], [2, 2, -3]]},
        {"_name": "negative weights", "n": 4, "edges": [[0, 1, -2], [1, 2, -3], [2, 3, 5], [0, 3, 1], [1, 3, 0]]},
        _connected_case(1, 30, 80, 0, 20, "random n=30, ties"),
        _connected_case(2, 2000, 6000, -1000, 1000, "random n=2000"),
    ]


def performance_case(seed: int = 0):
    return _connected_case(seed, 100_000, 300_000, 1, 1_000_000, "perf")


def build() -> Problem:
    return Problem(
        slug="kruskal_mst",
        title="Minimum spanning tree",
        difficulty="medium",
        summary="Kruskal's algorithm: sort the edges and join components with union-find.",
        params=["n", "edges"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["graphs", "union-find", "greedy", "sorting"],
    )
