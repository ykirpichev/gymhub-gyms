import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 200, "cpp": 6}


def reference(n, ops):
    parent = list(range(n))
    size = [1] * n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    answers = []
    for op, a, b in ops:
        ra, rb = find(a), find(b)
        if op == "same":
            answers.append(ra == rb)
        elif op == "union":
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
        else:
            raise ValueError(f"unknown op {op!r}")
    return answers


def _random_case(seed, n, n_ops, union_ratio, name):
    rng = random.Random(seed)
    ops = []
    for _ in range(n_ops):
        kind = "union" if rng.random() < union_ratio else "same"
        ops.append([kind, rng.randrange(n), rng.randrange(n)])
    return {"_name": name, "n": n, "ops": ops}


def _chain_case(seed, n, n_queries, name):
    # Unions that build one long chain when roots are linked naively, then queries
    # that start from the deep end of the chain.
    rng = random.Random(seed)
    ops = [["union", i, i + 1] for i in range(n - 1)]
    for _ in range(n_queries):
        if rng.random() < 0.1:
            ops.append(["union", rng.randrange(n), rng.randrange(n)])
        else:
            ops.append(["same", rng.randrange(n // 10), rng.randrange(n)])
    return {"_name": name, "n": n, "ops": ops}


def functional_cases():
    return [
        {
            "_name": "example",
            "n": 5,
            "ops": [["union", 0, 1], ["same", 0, 1], ["same", 1, 2], ["union", 1, 2],
                    ["same", 0, 2], ["same", 3, 4], ["union", 3, 4], ["same", 2, 4]],
        },
        {"_name": "no ops", "n": 3, "ops": []},
        {"_name": "single node", "n": 1, "ops": [["same", 0, 0], ["union", 0, 0], ["same", 0, 0]]},
        {"_name": "only queries", "n": 4, "ops": [["same", 0, 1], ["same", 2, 2], ["same", 3, 0]]},
        {
            "_name": "repeated unions",
            "n": 4,
            "ops": [["union", 0, 1], ["union", 1, 0], ["union", 0, 1], ["same", 1, 0], ["same", 0, 2]],
        },
        {
            "_name": "merging two groups",
            "n": 6,
            "ops": [["union", 0, 1], ["union", 1, 2], ["union", 3, 4], ["union", 4, 5],
                    ["same", 0, 5], ["union", 2, 3], ["same", 0, 5], ["same", 5, 1]],
        },
        _random_case(1, 30, 120, 0.3, "random n=30"),
        _chain_case(2, 500, 500, "long chain n=500"),
        _random_case(3, 5000, 20000, 0.2, "random n=5000"),
    ]


def performance_case(seed: int = 0):
    return _chain_case(seed, 100_000, 200_000, "perf")


def build() -> Problem:
    return Problem(
        slug="union_find",
        title="Union-find",
        difficulty="medium",
        summary="Disjoint-set union with path compression and union by size, answering connectivity queries.",
        params=["n", "ops"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["disjoint set", "union-find", "amortized analysis"],
    )
