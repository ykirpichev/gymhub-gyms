"""Grader for retry_with_idempotency. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Cases are JSON. Each op is ["run", key, steps, transient_name]. A step is
{"raise": name, "message": str} or {"return": value}, with an optional "effect" recorded
before the raise or return. The injected sleep appends its delay instead of waiting.
"""

import numpy as np

from gymkit.spec import Problem


class FlakyTimeout(TimeoutError):
    """Subclass of TimeoutError, so isinstance(..., TimeoutError) is true."""


_EXC = {
    "TimeoutError": TimeoutError,
    "ConnectionError": ConnectionError,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "FlakyTimeout": FlakyTimeout,
}


def _operation(key, steps, effects, executions):
    cursor = {"i": 0}

    def fn():
        step = steps[cursor["i"]]
        cursor["i"] += 1
        executions.append(key)
        if "effect" in step:
            effects.append(step["effect"])
        if "raise" in step:
            raise _EXC[step["raise"]](step["message"])
        return step["return"]

    return fn


def _drive(cls, base, cap, max_attempts, ops):
    sleeps, effects, executions = [], [], []

    def sleep(delay):
        sleeps.append(delay)

    executor = cls(sleep, base, cap, max_attempts)
    results = []
    for op in ops:
        _, key, steps, transient_name = op
        fn = _operation(key, steps, effects, executions)
        results.append(executor.run(key, fn, _EXC[transient_name]))
    return {"results": results, "sleeps": sleeps, "effects": effects, "executions": executions}


class _Reference:
    def __init__(self, sleep, base, cap, max_attempts):
        self.sleep = sleep
        self.base = base
        self.cap = cap
        self.max_attempts = max_attempts
        self.stored = {}

    def run(self, key, fn, transient):
        if key in self.stored:
            return self.stored[key]
        delays = []
        for attempt in range(self.max_attempts):
            try:
                value = fn()
            except transient as exc:
                if attempt + 1 == self.max_attempts:
                    result = _failure(exc, attempt + 1, delays)
                    self.stored[key] = result
                    return result
                delay = min(self.cap, self.base * 2 ** attempt)
                delays.append(delay)
                self.sleep(delay)
            except Exception as exc:
                result = _failure(exc, attempt + 1, delays)
                self.stored[key] = result
                return result
            else:
                result = {"ok": True, "value": value, "attempts": attempt + 1, "delays": delays}
                self.stored[key] = result
                return result


def _failure(exc, attempts, delays):
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "attempts": attempts, "delays": delays}


def reference(base, cap, max_attempts, ops):
    return _drive(_Reference, base, cap, max_attempts, ops)


def call(module, args):
    if not hasattr(module, "IdempotentExecutor"):
        raise AttributeError("the solution does not define class IdempotentExecutor")
    return _drive(module.IdempotentExecutor, *args)


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, dict):
        raise AssertionError(f"internal result should be a dict, got {type(got).__name__}: {got!r:.200}")
    labels = (
        ("effects", "side effects recorded by the operation (a repeated key must not record another)"),
        ("executions", "calls to fn, one entry per execution"),
        ("sleeps", "delays passed to sleep, in order"),
        ("results", "dicts returned by run"),
    )
    for key, label in labels:
        try:
            default_compare(got.get(key), want[key], 1e-6, 1e-6, label)
        except AssertionError as exc:
            raise AssertionError(str(exc)) from None


def _run(key, steps, transient="TimeoutError"):
    return ["run", key, steps, transient]


def _random(seed, n_keys, name):
    rng = np.random.default_rng(seed)
    max_attempts = 3
    ops = []
    for i in range(n_keys):
        steps = []
        for j in range(max_attempts):
            roll = int(rng.integers(0, 3))
            if roll == 0:
                steps.append({"raise": "TimeoutError", "message": f"t{i}-{j}"})
            elif roll == 1:
                steps.append({"raise": "ValueError", "message": f"p{i}-{j}"})
            else:
                steps.append({"effect": f"e{i}", "return": int(rng.integers(0, 20))})
        steps.append({"effect": "extra", "return": 999})
        ops.append(_run(f"k{i}", steps))
    for i in range(n_keys):
        if float(rng.random()) < 0.6:
            ops.append(_run(f"k{i}", [{"effect": "replay", "return": -1}], "ValueError"))
    return {"_name": name, "base": 1, "cap": 4, "max_attempts": max_attempts, "ops": ops}


def functional_cases():
    return [
        {"_name": "example", "base": 1, "cap": 5, "max_attempts": 4, "ops": [
            _run("charge", [
                {"raise": "TimeoutError", "message": "slow"},
                {"raise": "TimeoutError", "message": "slow"},
                {"effect": "charge", "return": {"id": 7}},
            ]),
            _run("charge", [{"effect": "charge", "return": {"id": 8}}], "ConnectionError")]},
        {"_name": "success on the first try does not sleep", "base": 4, "cap": 4, "max_attempts": 3, "ops": [
            _run("job", [{"effect": "once", "return": 0}]),
            _run("job", [{"effect": "twice", "return": 1}])]},
        {"_name": "a permanent error is not retried", "base": 1, "cap": 8, "max_attempts": 4, "ops": [
            _run("job", [
                {"raise": "ValueError", "message": "bad"},
                {"effect": "should-not", "return": 1},
            ])]},
        {"_name": "a subclass of the transient class is retried", "base": 2, "cap": 10, "max_attempts": 3, "ops": [
            _run("job", [
                {"raise": "FlakyTimeout", "message": "slow"},
                {"return": "recovered"},
            ])]},
        {"_name": "a sibling exception is permanent", "base": 1, "cap": 8, "max_attempts": 3, "ops": [
            _run("job", [
                {"raise": "ConnectionError", "message": "refused"},
                {"return": "nope"},
            ])]},
        {"_name": "the attempt budget stops on the last transient error", "base": 1, "cap": 8, "max_attempts": 3, "ops": [
            _run("job", [
                {"raise": "TimeoutError", "message": "a"},
                {"raise": "TimeoutError", "message": "b"},
                {"raise": "TimeoutError", "message": "c"},
                {"effect": "late", "return": "should-not"},
            ])]},
        {"_name": "the cap clamps the delay", "base": 3, "cap": 10, "max_attempts": 5, "ops": [
            _run("job", [
                {"raise": "TimeoutError", "message": "t0"},
                {"raise": "TimeoutError", "message": "t1"},
                {"raise": "TimeoutError", "message": "t2"},
                {"raise": "TimeoutError", "message": "t3"},
                {"return": "ok"},
            ])]},
        {"_name": "one attempt means no retry and no sleep", "base": 1, "cap": 10, "max_attempts": 1, "ops": [
            _run("job", [
                {"raise": "TimeoutError", "message": "slow"},
                {"return": "should-not"},
            ])]},
        {"_name": "keys are independent", "base": 1, "cap": 10, "max_attempts": 3, "ops": [
            _run("a", [
                {"raise": "TimeoutError", "message": "t"},
                {"effect": "a", "return": 1},
            ]),
            _run("b", [{"effect": "b", "return": 2}]),
            _run("a", [{"effect": "a", "return": 9}])]},
        {"_name": "a failed run is stored too", "base": 1, "cap": 8, "max_attempts": 2, "ops": [
            _run("job", [
                {"raise": "TimeoutError", "message": "one"},
                {"raise": "TimeoutError", "message": "two"},
                {"return": "later"},
            ]),
            _run("job", [{"return": "later"}])]},
        {"_name": "a zero delay is still passed to sleep", "base": 0, "cap": 4, "max_attempts": 4, "ops": [
            _run("job", [
                {"raise": "TimeoutError", "message": "t"},
                {"raise": "TimeoutError", "message": "t"},
                {"return": 5},
            ])]},
        {"_name": "permanent error after a transient one", "base": 1, "cap": 10, "max_attempts": 4, "ops": [
            _run("job", [
                {"raise": "TimeoutError", "message": "t"},
                {"effect": "wrote", "raise": "ValueError", "message": "bad"},
                {"effect": "again", "return": 1},
            ])]},
        _random(1, 6, "random 6 keys"),
        _random(2, 10, "random 10 keys"),
    ]


def build() -> Problem:
    return Problem(
        slug="retry_with_idempotency",
        title="Retry with idempotency",
        difficulty="medium",
        summary="Retry a transient exception with capped exponential backoff, and return the stored result for a repeated idempotency key.",
        params=["base", "cap", "max_attempts", "ops"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        atol=1e-6,
        rtol=1e-6,
        tags=["agents", "retries", "idempotency"],
    )
