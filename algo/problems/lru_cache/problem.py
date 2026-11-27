import random
from collections import OrderedDict

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 300, "cpp": 50}


class _ReferenceLRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def get(self, key):
        if key not in self.data:
            return -1
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)


def _drive(cls, capacity, ops):
    cache = cls(capacity)
    results = []
    for op in ops:
        if op[0] == "get":
            results.append(cache.get(op[1]))
        elif op[0] == "put":
            cache.put(op[1], op[2])
        else:
            raise ValueError(f"unknown op {op[0]!r}")
    return results


def reference(capacity, ops):
    return _drive(_ReferenceLRU, capacity, ops)


def call(module, args):
    if not hasattr(module, "LRUCache"):
        raise AttributeError("the solution does not define class LRUCache")
    return _drive(module.LRUCache, *args)


def compare(got, want, case):
    if not isinstance(got, list) or len(got) != len(want):
        raise AssertionError(f"expected {len(want)} get results, got {got!r:.120}")
    gets = [op for op in case["ops"] if op[0] == "get"]
    for index, (g, w) in enumerate(zip(got, want)):
        if g != w:
            position = [i for i, op in enumerate(case["ops"]) if op[0] == "get"][index]
            raise AssertionError(
                f"get #{index} (ops[{position}] = {gets[index]!r}) returned {g!r}, expected {w!r}"
            )


def _random_case(seed, capacity, n_ops, key_space, get_ratio, name):
    rng = random.Random(seed)
    ops = []
    for _ in range(n_ops):
        key = rng.randrange(key_space)
        if rng.random() < get_ratio:
            ops.append(["get", key])
        else:
            ops.append(["put", key, rng.randrange(10**6)])
    return {"_name": name, "capacity": capacity, "ops": ops}


def functional_cases():
    return [
        {
            "_name": "example",
            "capacity": 2,
            "ops": [["put", 1, 10], ["put", 2, 20], ["get", 1], ["put", 3, 30], ["get", 2],
                    ["put", 4, 40], ["get", 1], ["get", 3], ["get", 4]],
        },
        {"_name": "no ops", "capacity": 3, "ops": []},
        {"_name": "get on empty cache", "capacity": 2, "ops": [["get", 5], ["get", 0]]},
        {
            "_name": "capacity 1",
            "capacity": 1,
            "ops": [["put", 1, 1], ["put", 2, 2], ["get", 1], ["get", 2], ["put", 2, 7], ["get", 2]],
        },
        {
            "_name": "put on existing key refreshes it",
            "capacity": 2,
            "ops": [["put", 1, 1], ["put", 2, 2], ["put", 1, 100], ["put", 3, 3], ["get", 1], ["get", 2], ["get", 3]],
        },
        {
            "_name": "get refreshes recency",
            "capacity": 3,
            "ops": [["put", "a", 1], ["put", "b", 2], ["put", "c", 3], ["get", "a"], ["put", "d", 4],
                    ["get", "b"], ["get", "a"], ["get", "c"], ["get", "d"]],
        },
        {
            "_name": "value 0 is a hit, not a miss",
            "capacity": 2,
            "ops": [["put", 9, 0], ["get", 9], ["put", 8, 0], ["get", 8], ["get", 9]],
        },
        _random_case(1, 5, 300, 12, 0.5, "random capacity=5"),
        _random_case(2, 100, 5000, 300, 0.6, "random capacity=100"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 50_000, 300_000, 100_000, 0.5, "perf")


def build() -> Problem:
    return Problem(
        slug="lru_cache",
        title="LRU cache",
        difficulty="medium",
        summary="Design a fixed-capacity cache with O(1) get and put that evicts the least recently used key.",
        params=["capacity", "ops"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        compare=compare,
        call=call,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["class design", "hash map", "linked list"],
    )
