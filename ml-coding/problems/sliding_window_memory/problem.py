"""Grader for sliding_window_memory. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Cases are JSON. `counter` and `summarizer` name pure functions injected into the class.
An op is ["build", messages]. ValueError becomes the marker "ValueError".
"""

import numpy as np

from gymkit.spec import Problem


def _count_field(message):
    return message["tokens"]


def _count_plus(message):
    return message["tokens"] + 1


def _summarize_growing(evicted):
    return {
        "role": "system",
        "pinned": False,
        "content": "summary:" + "|".join(message["content"] for message in evicted),
        "tokens": 1 + len(evicted),
    }


def _summarize_fixed(evicted):
    return {
        "role": "system",
        "pinned": False,
        "content": "summary:" + "|".join(message["content"] for message in evicted),
        "tokens": 4,
    }


_COUNTERS = {"field": _count_field, "plus": _count_plus}
_SUMMARIZERS = {"growing": _summarize_growing, "fixed": _summarize_fixed}


class _Memory:
    def __init__(self, count_tokens, summarize, budget):
        self.count_tokens = count_tokens
        self.summarize = summarize
        self.budget = budget

    def build(self, messages):
        pinned = [message for message in messages if message["pinned"]]
        rest = [message for message in messages if not message["pinned"]]
        pinned_cost = sum(self.count_tokens(message) for message in pinned)
        if pinned_cost > self.budget:
            raise ValueError("pinned messages exceed the token budget")
        room = self.budget - pinned_cost
        costs = [self.count_tokens(message) for message in rest]
        if sum(costs) <= room:
            return pinned + rest
        for k in range(1, len(rest) + 1):
            summary = self.summarize(rest[:k])
            if self.count_tokens(summary) + sum(costs[k:]) <= room:
                return pinned + [summary] + rest[k:]
        raise ValueError("no summary of the evicted messages fits in the budget")


def _msg(role, content, pinned, tokens):
    return {"role": role, "content": content, "pinned": pinned, "tokens": tokens}


def _drive(cls, budget, counter, summarizer, ops):
    memory = cls(_COUNTERS[counter], _SUMMARIZERS[summarizer], budget)
    results = []
    for op in ops:
        try:
            results.append(memory.build(op[1]))
        except ValueError:
            results.append("ValueError")
    return results


def reference(budget, counter, summarizer, ops):
    return _drive(_Memory, budget, counter, summarizer, ops)


def call(module, args):
    if not hasattr(module, "SlidingWindowMemory"):
        raise AttributeError("the solution does not define class SlidingWindowMemory")
    return _drive(module.SlidingWindowMemory, *args)


def compare(got, want, case):
    if not isinstance(got, list) or len(got) != len(want):
        kind = type(got).__name__
        length = len(got) if isinstance(got, list) else "?"
        raise AssertionError(f"expected {len(want)} build results, got {kind} of length {length}: {got!r:.240}")
    for index, (g, w) in enumerate(zip(got, want)):
        if w == "ValueError" and g != "ValueError":
            raise AssertionError(
                f"ops[{index}] should raise ValueError (pinned messages over budget, or no summary fits), got {g!r:.360}"
            )
        if g == "ValueError" and w != "ValueError":
            raise AssertionError(f"ops[{index}] raised ValueError\n  expected: {w!r:.360}")
        if g != w:
            raise AssertionError(f"ops[{index}]:\n  got:      {g!r:.500}\n  expected: {w!r:.500}")


_EXAMPLE = [
    _msg("system", "rules", True, 3),
    _msg("user", "ancient", False, 4),
    _msg("assistant", "ancient-reply", False, 4),
    _msg("system", "format", True, 2),
    _msg("user", "recent", False, 3),
    _msg("assistant", "recent-reply", False, 2),
]


def _random(seed, n, budget, summarizer, name):
    rng = np.random.default_rng(seed)
    roles = ["system", "user", "assistant"]
    messages = []
    for i in range(n):
        messages.append(_msg(
            roles[int(rng.integers(0, 3))],
            f"m{i}",
            bool(float(rng.random()) < 0.25),
            int(rng.integers(1, 6)),
        ))
    return {
        "_name": name,
        "budget": budget,
        "counter": "field",
        "summarizer": summarizer,
        "ops": [["build", messages]],
    }


def functional_cases():
    four = [_msg("user", name, False, 3) for name in ("w", "x", "y", "z")]
    return [
        {"_name": "example", "budget": 12, "counter": "field", "summarizer": "growing",
         "ops": [["build", _EXAMPLE]]},
        {"_name": "exact fit keeps every message", "budget": 8, "counter": "field", "summarizer": "growing",
         "ops": [["build", [
             _msg("system", "pin", True, 2),
             _msg("user", "a", False, 3),
             _msg("assistant", "b", False, 3)]]],},
        {"_name": "pinned messages only", "budget": 10, "counter": "field", "summarizer": "growing",
         "ops": [["build", [_msg("system", "a", True, 3), _msg("system", "b", True, 4)]]],},
        {"_name": "pinned messages over the budget", "budget": 4, "counter": "field", "summarizer": "growing",
         "ops": [["build", [_msg("system", "rules", True, 5), _msg("user", "hi", False, 1)]]],},
        {"_name": "the summary's tokens force a deeper cut", "budget": 10, "counter": "field",
         "summarizer": "growing", "ops": [["build", four]]},
        {"_name": "summary plus kept messages may fill the room exactly", "budget": 10,
         "counter": "field", "summarizer": "growing", "ops": [["build", [
             _msg("user", "p", False, 3),
             _msg("user", "q", False, 4),
             _msg("user", "r", False, 4)]]],},
        {"_name": "everything unpinned is evicted", "budget": 5, "counter": "field", "summarizer": "growing",
         "ops": [["build", [_msg("user", "a", False, 4), _msg("user", "b", False, 4)]]],},
        {"_name": "pinned messages fill the budget", "budget": 3, "counter": "field", "summarizer": "growing",
         "ops": [["build", [_msg("system", "rules", True, 3), _msg("user", "more", False, 2)]]],},
        {"_name": "a recent message that cannot sit beside the summary is dropped", "budget": 6,
         "counter": "field", "summarizer": "growing", "ops": [["build", [
             _msg("user", "old", False, 1),
             _msg("user", "new", False, 6)]]],},
        {"_name": "a fixed-cost summary evicts more", "budget": 12, "counter": "field", "summarizer": "fixed",
         "ops": [["build", [
             _msg("user", "a", False, 5),
             _msg("user", "b", False, 5),
             _msg("user", "c", False, 5)]]],},
        {"_name": "count_tokens is the injected function", "budget": 8, "counter": "plus",
         "summarizer": "growing", "ops": [["build", [
             _msg("user", "a", False, 4),
             _msg("user", "b", False, 4)]]],},
        {"_name": "builds do not share state", "budget": 10, "counter": "field", "summarizer": "growing",
         "ops": [
             ["build", four],
             ["build", []],
             ["build", [_msg("system", "only", True, 2)]]],},
        _random(1, 20, 18, "growing", "random 20 messages"),
        _random(2, 32, 15, "fixed", "random 32 messages"),
    ]


def build() -> Problem:
    return Problem(
        slug="sliding_window_memory",
        title="Sliding-window conversation memory",
        difficulty="medium",
        summary="Keep pinned messages and as much recent history as fits, replacing evicted messages with one summary.",
        params=["budget", "counter", "summarizer", "ops"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        atol=1e-6,
        rtol=1e-6,
        tags=["agents", "memory", "context window"],
    )
