"""Grader for kv_cache. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Ops are ["prefill", K, V], ["append", k, v], ["get"], ["attention", query],
or ["memory", layers, heads, dim, seq, batch, dtype]. A ValueError stops the script.
"""

import numpy as np

from gymkit.spec import Problem


class _KV:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive int")
        self.capacity = capacity
        self.k = None
        self.v = None

    def prefill(self, K, V):
        if self.k is not None:
            raise ValueError("prefill requires an empty cache")
        keys = np.asarray(K, dtype=np.float64)
        values = np.asarray(V, dtype=np.float64)
        if keys.ndim != 3 or keys.shape != values.shape:
            raise ValueError("K and V must share shape (heads, T, d)")
        heads, steps, dim = keys.shape
        if heads < 1 or dim < 1 or steps < 1 or steps > self.capacity:
            raise ValueError("prefill exceeds capacity or is empty")
        self.k = keys.copy()
        self.v = values.copy()

    def append(self, k, v):
        key = np.asarray(k, dtype=np.float64)
        value = np.asarray(v, dtype=np.float64)
        if key.ndim != 2 or key.shape != value.shape:
            raise ValueError("k and v must share shape (heads, d)")
        heads, dim = key.shape
        if heads < 1 or dim < 1:
            raise ValueError("k and v must have shape (heads, d)")
        if self.k is None:
            self.k = key[:, None, :].copy()
            self.v = value[:, None, :].copy()
            return
        if (heads, dim) != (self.k.shape[0], self.k.shape[2]):
            raise ValueError("k and v must match the cache heads and d")
        if self.k.shape[1] + 1 > self.capacity:
            raise ValueError("capacity exceeded")
        self.k = np.concatenate([self.k, key[:, None, :]], axis=1)
        self.v = np.concatenate([self.v, value[:, None, :]], axis=1)

    def get(self):
        if self.k is None:
            raise ValueError("empty cache")
        return self.k.copy(), self.v.copy()

    def attention(self, query):
        if self.k is None:
            raise ValueError("empty cache")
        q = np.asarray(query, dtype=np.float64)
        heads, _steps, dim = self.k.shape
        if q.shape != (heads, dim):
            raise ValueError("query must have shape (heads, d)")
        scores = np.einsum("hd,htd->ht", q, self.k) / np.sqrt(dim)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        return np.einsum("ht,htd->hd", weights, self.v)

    @staticmethod
    def memory_bytes(layers, heads, dim, seq, batch, dtype):
        dims = (layers, heads, dim, seq, batch)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in dims):
            raise ValueError("layers, heads, dim, seq, and batch must be non-negative ints")
        try:
            itemsize = int(np.dtype(dtype).itemsize)
        except TypeError:
            raise ValueError(f"unknown dtype {dtype!r}") from None
        return 2 * layers * heads * dim * seq * batch * itemsize


def _one(cache, op):
    kind = op[0]
    if kind == "prefill":
        cache.prefill(op[1], op[2])
        return None
    if kind == "append":
        cache.append(op[1], op[2])
        return None
    if kind == "get":
        return cache.get()
    if kind == "attention":
        return cache.attention(op[1])
    if kind == "memory":
        return cache.memory_bytes(*op[1:])
    raise RuntimeError(f"unknown op {kind!r}")


def _drive(cls, capacity, ops):
    try:
        cache = cls(capacity)
    except ValueError:
        return ["ValueError"]
    results = []
    for op in ops:
        try:
            results.append(_one(cache, op))
        except ValueError:
            results.append("ValueError")
            break
    return results


def reference(capacity, ops):
    return _drive(_KV, capacity, ops)


def call(module, args):
    if not hasattr(module, "KVCache"):
        raise AttributeError("the solution does not define class KVCache")
    return _drive(module.KVCache, *args)


def _has_nan(value):
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return False
    if isinstance(value, tuple):
        return any(_has_nan(part) for part in value)
    array = np.asarray(value)
    return array.dtype.kind == "f" and bool(np.isnan(array).any())


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, list) or len(got) != len(want):
        got_n = len(got) if isinstance(got, list) else type(got).__name__
        raise AssertionError(f"expected {len(want)} results, got {got_n}")
    for index, (g, w) in enumerate(zip(got, want)):
        if index < len(case["ops"]):
            where = f"ops[{index}] {case['ops'][index][0]}"
        else:
            where = "KVCache()"
        got_err = isinstance(g, str) and g == "ValueError"
        want_err = isinstance(w, str) and w == "ValueError"
        if got_err or want_err:
            if got_err != want_err:
                if want_err:
                    raise AssertionError(f"{where}: expected ValueError")
                raise AssertionError(f"{where}: raised ValueError")
            continue
        if _has_nan(g):
            raise AssertionError(f"{where}: output contains NaN (softmax without subtracting the max?)")
        default_compare(g, w, 1e-9, 1e-7, where)


def _sequence(seed, heads, steps, dim, name, prefill):
    rng = np.random.default_rng(seed)
    queries = rng.standard_normal((heads, steps, dim))
    keys = rng.standard_normal((heads, steps, dim))
    values = rng.standard_normal((heads, steps, dim))
    ops = []
    start = 0
    if prefill:
        ops.append(["prefill", keys[:, :prefill].tolist(), values[:, :prefill].tolist()])
        ops.append(["attention", queries[:, prefill - 1].tolist()])
        start = prefill
    for t in range(start, steps):
        ops.append(["append", keys[:, t].tolist(), values[:, t].tolist()])
        ops.append(["attention", queries[:, t].tolist()])
    return {"_name": name, "capacity": steps, "ops": ops}


def functional_cases():
    keys = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]
    values = [[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]
    return [
        {"_name": "example", "capacity": 4, "ops": [
            ["prefill", keys, values],
            ["attention", [[1.0, 0.0, 0.0, 0.0]]],
            ["append", [[0.0, 0.0, 1.0, 0.0]], [[0.0, 0.0, 0.0, 4.0]]],
            ["get"],
            ["attention", [[0.0, 0.0, 1.0, 0.0]]],
            ["memory", 1, 1, 4, 8, 1, "float32"],
        ]},
        _sequence(1, 2, 5, 4, "decode token by token", 0),
        _sequence(2, 2, 7, 4, "prefill then decode", 3),
        _sequence(3, 4, 24, 8, "random 4 heads, 24 steps", 0),
        {"_name": "huge scores stay finite", "capacity": 2, "ops": [
            ["prefill", [[[1000.0], [0.0]]], [[[5.0], [7.0]]]],
            ["attention", [[1000.0]]],
        ]},
        {"_name": "append past capacity", "capacity": 2, "ops": [
            ["append", [[1.0]], [[4.0]]],
            ["append", [[2.0]], [[5.0]]],
            ["attention", [[1.0]]],
            ["append", [[3.0]], [[6.0]]],
        ]},
        {"_name": "prefill longer than capacity", "capacity": 2, "ops": [
            ["prefill", [[[1.0], [2.0], [3.0]]], [[[4.0], [5.0], [6.0]]]],
        ]},
        {"_name": "second prefill", "capacity": 4, "ops": [
            ["prefill", [[[1.0, 0.0]]], [[[0.0, 1.0]]]],
            ["prefill", [[[2.0, 0.0]]], [[[0.0, 2.0]]]],
        ]},
        {"_name": "K and V shapes differ", "capacity": 2, "ops": [
            ["prefill", [[[1.0, 2.0]]], [[[1.0, 2.0, 3.0]]]],
        ]},
        {"_name": "append shape mismatch", "capacity": 3, "ops": [
            ["prefill", [[[1.0, 2.0]]], [[[3.0, 4.0]]]],
            ["append", [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]],
        ]},
        {"_name": "get on an empty cache", "capacity": 2, "ops": [["get"]]},
        {"_name": "attention on an empty cache", "capacity": 2, "ops": [["attention", [[1.0, 0.0]]]]},
        {"_name": "memory bytes by dtype", "capacity": 1, "ops": [
            ["memory", 2, 4, 8, 16, 2, "float32"],
            ["memory", 2, 4, 8, 16, 2, "float16"],
            ["memory", 2, 4, 8, 16, 2, "float64"],
        ]},
        {"_name": "unknown dtype", "capacity": 1, "ops": [
            ["memory", 1, 1, 1, 1, 1, "float32"],
            ["memory", 1, 1, 1, 1, 1, "not-a-dtype"],
        ]},
        {"_name": "capacity 0", "capacity": 0, "ops": []},
    ]


def build() -> Problem:
    return Problem(
        slug="kv_cache",
        title="Append-only KV cache",
        difficulty="medium",
        summary="Fixed-capacity keys and values, with prefill, one-step append, scaled attention, and a memory estimate.",
        params=["capacity", "ops"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "attention", "inference"],
    )
