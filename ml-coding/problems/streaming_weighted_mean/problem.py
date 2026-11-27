"""Grader for streaming_weighted_mean. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Ops drive StreamingWeightedMean the way LRUCache is driven. Merging shard state must match one
pass over the same samples. Averaging per-batch means fails when batch weights differ.
"""

import numpy as np

from gymkit.spec import Problem


class _ReferenceMean:
    def __init__(self):
        self.count = 0
        self.weight_sum = 0.0
        self.weighted_sum = 0.0

    def update(self, values, weights):
        if len(values) != len(weights):
            raise ValueError("values and weights must have the same length")
        batch = []
        for value, weight in zip(values, weights):
            value = float(value)
            weight = float(weight)
            if weight < 0:
                raise ValueError("weights must be non-negative")
            batch.append((value, weight))
        for value, weight in batch:
            self.count += 1
            self.weight_sum += weight
            self.weighted_sum += weight * value

    def merge(self, other):
        self.count += other.count
        self.weight_sum += other.weight_sum
        self.weighted_sum += other.weighted_sum

    def result(self):
        if self.weight_sum == 0.0:
            return None
        return self.weighted_sum / self.weight_sum


def _step(model, op):
    kind = op[0]
    if kind == "update":
        model.update(op[1], op[2])
    elif kind == "merge":
        other = model.__class__()
        for inner in op[1]:
            _step(other, inner)
        model.merge(other)
    elif kind == "result":
        return
    else:
        raise ValueError(f"unknown op {kind!r}")


def _drive(cls, ops):
    model = cls()
    results = []
    for op in ops:
        try:
            _step(model, op)
        except ValueError:
            results.append("ValueError")
            continue
        if op[0] == "result":
            results.append(model.result())
    return results


def reference(ops):
    return _drive(_ReferenceMean, ops)


def call(module, args):
    if not hasattr(module, "StreamingWeightedMean"):
        raise AttributeError("the solution does not define class StreamingWeightedMean")
    return _drive(module.StreamingWeightedMean, args[0])


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, list) or len(got) != len(want):
        raise AssertionError(f"expected {len(want) if isinstance(want, list) else '?'} recorded results, got {got!r}")
    for index, (g, w) in enumerate(zip(got, want)):
        if w is None or w == "ValueError":
            if g != w:
                detail = "a negative weight or a length mismatch must raise ValueError and leave the totals unchanged" if w == "ValueError" else "an empty total weight is None, not 0"
                raise AssertionError(f"result {index}: got {g!r}, expected {w!r}. {detail}")
        else:
            try:
                default_compare(g, w, 1e-8, 1e-6, f"result {index}")
            except AssertionError as exc:
                raise AssertionError(str(exc) + "\n  add up weight * value and weight; do not average per-batch means") from None


def _random_ops(seed, batches, name):
    rng = np.random.default_rng(seed)
    ops = []
    shards = []
    for _ in range(batches):
        width = int(rng.integers(1, 5))
        values = rng.normal(size=width).tolist()
        weights = rng.uniform(0.1, 3.0, size=width).tolist()
        if rng.random() < 0.3:
            weights[0] = 0.0
        shards.append(["update", values, weights])
        if len(shards) == 2 or rng.random() < 0.4:
            ops.append(["merge", shards])
            shards = []
            if rng.random() < 0.5:
                ops.append(["result"])
    if shards:
        ops.append(["merge", shards])
    ops.append(["result"])
    return {"_name": name, "ops": ops}


def functional_cases():
    return [
        {
            "_name": "example",
            "ops": [
                ["update", [0.0, 0.0], [1.0, 3.0]],
                ["update", [10.0], [1.0]],
                ["result"],
            ],
        },
        {
            "_name": "running mean keeps every batch",
            "ops": [
                ["update", [1.0], [1.0]],
                ["result"],
                ["update", [3.0], [1.0]],
                ["result"],
                ["update", [8.0], [2.0]],
                ["result"],
            ],
        },
        {"_name": "empty", "ops": [["result"]]},
        {"_name": "empty batch stays empty", "ops": [["update", [], []], ["result"]]},
        {"_name": "all zero weights", "ops": [["update", [5.0, 8.0], [0.0, 0.0]], ["result"]]},
        {
            "_name": "zero weight then a real one",
            "ops": [["update", [5.0], [0.0]], ["update", [8.0], [2.0]], ["result"]],
        },
        {
            "_name": "negative weight does not change state",
            "ops": [
                ["update", [4.0, 6.0], [1.0, 1.0]],
                ["result"],
                ["update", [100.0], [-0.5]],
                ["result"],
            ],
        },
        {"_name": "length mismatch", "ops": [["update", [1.0, 2.0], [1.0]], ["result"]]},
        {
            "_name": "merge matches one pass",
            "ops": [
                ["update", [0.0, 0.0], [1.0, 3.0]],
                ["merge", [["update", [10.0], [1.0]]]],
                ["result"],
            ],
        },
        {
            "_name": "three shards",
            "ops": [
                ["merge", [["update", [1.0], [1.0]]]],
                ["merge", [["update", [5.0], [1.0]], ["update", [5.0], [2.0]]]],
                ["merge", [["update", [0.0], [4.0]]]],
                ["result"],
            ],
        },
        {"_name": "merge empty shard", "ops": [["update", [2.0], [5.0]], ["merge", []], ["result"]]},
        {"_name": "negative values", "ops": [["update", [-2.0, 4.0], [1.0, 1.0]], ["result"]]},
        _random_ops(11, 6, "random six batches merged"),
        _random_ops(12, 12, "random twelve batches merged"),
    ]


def build() -> Problem:
    return Problem(
        slug="streaming_weighted_mean",
        title="Streaming weighted mean",
        difficulty="medium",
        summary="Aggregate a weighted mean by count, weight sum and weighted sum so shards merge exactly.",
        params=["ops"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["streaming", "evaluation", "weighted mean"],
    )
