import heapq
import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 1200, "cpp": 70}


def reference(n, edges, src):
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    dist = [-1] * n
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if dist[u] != -1:
            continue
        dist[u] = d
        for v, w in graph[u]:
            if dist[v] == -1:
                heapq.heappush(heap, (d + w, v))
    return dist


def _random_case(seed, n, m, max_w, name):
    rng = random.Random(seed)
    edges = [[rng.randrange(n), rng.randrange(n), rng.randint(0, max_w)] for _ in range(m)]
    return {"_name": name, "n": n, "edges": edges, "src": rng.randrange(n)}


def _perf_graph(seed, n, m, max_w, name):
    # A random spanning tree out of node 0 keeps (almost) everything reachable,
    # plus random extra edges that create many competing paths.
    rng = random.Random(seed)
    order = list(range(1, n))
    rng.shuffle(order)
    order = [0] + order
    edges = []
    for index in range(1, n):
        edges.append([order[rng.randrange(index)], order[index], rng.randint(1, max_w)])
    for _ in range(m - (n - 1)):
        edges.append([rng.randrange(n), rng.randrange(n), rng.randint(0, max_w)])
    rng.shuffle(edges)
    return {"_name": name, "n": n, "edges": edges, "src": 0}


def functional_cases():
    return [
        {
            "_name": "example",
            "n": 5,
            "edges": [[0, 1, 4], [0, 2, 1], [2, 1, 2], [1, 3, 1], [2, 3, 5]],
            "src": 0,
        },
        {"_name": "single node, no edges", "n": 1, "edges": [], "src": 0},
        {"_name": "no edges", "n": 4, "edges": [], "src": 2},
        {"_name": "directed, not undirected", "n": 3, "edges": [[1, 0, 1], [1, 2, 1]], "src": 0},
        {
            "_name": "zero weights and a self loop",
            "n": 4,
            "edges": [[0, 1, 0], [1, 2, 0], [2, 2, 5], [2, 3, 7], [0, 3, 8]],
            "src": 0,
        },
        {
            "_name": "parallel edges",
            "n": 3,
            "edges": [[0, 1, 9], [0, 1, 3], [0, 1, 6], [1, 2, 2], [0, 2, 10]],
            "src": 0,
        },
        {
            "_name": "greedy first edge is wrong",
            "n": 4,
            "edges": [[0, 1, 1], [1, 3, 100], [0, 2, 5], [2, 3, 5]],
            "src": 0,
        },
        _random_case(1, 30, 60, 20, "random n=30 m=60"),
        _random_case(2, 2000, 8000, 1000, "random n=2000 m=8000"),
        _perf_graph(3, 5000, 20000, 10**6, "tree plus extras n=5000"),
    ]


def performance_case(seed: int = 0):
    return _perf_graph(seed, 100_000, 300_000, 10**6, "perf")


def build() -> Problem:
    return Problem(
        slug="dijkstra",
        title="Dijkstra shortest paths",
        difficulty="medium",
        summary="Single-source shortest paths on a directed graph with non-negative weights using a binary heap.",
        params=["n", "edges", "src"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["graphs", "shortest paths", "heap"],
    )
