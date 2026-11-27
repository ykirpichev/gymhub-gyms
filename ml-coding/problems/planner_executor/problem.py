"""Grader for planner_executor. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Cases are JSON. `tools` is a list of fixture names; each run builds fresh functions that log
every call, including calls that then raise. `solve` sees the real dict of functions.
"""

import re

import numpy as np

from gymkit.spec import Problem

_REF = re.compile(r"\$([A-Za-z0-9_]+)\.output$")


def _refs(value):
    if isinstance(value, str):
        match = _REF.fullmatch(value)
        return [match.group(1)] if match else []
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _refs(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _refs(item)]
    return []


def _resolve(value, outputs):
    if isinstance(value, str):
        match = _REF.fullmatch(value)
        return outputs[match.group(1)] if match else value
    if isinstance(value, dict):
        return {key: _resolve(item, outputs) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, outputs) for item in value]
    return value


def _execute(plan, tools, max_steps):
    if len(plan) > max_steps:
        raise ValueError(f"plan has {len(plan)} steps, the limit is {max_steps}")
    seen = set()
    for step in plan:
        if step["id"] in seen:
            raise ValueError(f"duplicate step id {step['id']!r}")
        missing = [ref for ref in _refs(step["args"]) if ref not in seen]
        if missing:
            raise ValueError(f"step {step['id']!r} references {missing[0]!r}, which is not an earlier step")
        seen.add(step["id"])

    trace, status, outputs = [], {}, {}
    for step in plan:
        entry = {"id": step["id"], "tool": step["tool"]}
        blocked = sorted({ref for ref in _refs(step["args"]) if status[ref] != "ok"})
        if blocked:
            entry.update(status="skipped", blocked_by=blocked)
        elif step["tool"] not in tools:
            entry.update(status="error", error=f"unknown tool: {step['tool']}")
        else:
            try:
                output = tools[step["tool"]](**_resolve(step["args"], outputs))
            except Exception as exc:
                entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            else:
                entry.update(status="ok", output=output)
                outputs[step["id"]] = output
        status[step["id"]] = entry["status"]
        trace.append(entry)
    return trace


def _fixtures(log):
    pages = {"a.com": "alpha beta", "b.com": "one", "d.com": "x y z"}

    def fetch(url):
        log.append(["fetch", {"url": url}])
        if url not in pages:
            raise LookupError(f"404 {url}")
        return pages[url]

    def words(text):
        log.append(["words", {"text": text}])
        return text.split()

    def count(items):
        log.append(["count", {"items": items}])
        return len(items)

    def add(a, b):
        log.append(["add", {"a": a, "b": b}])
        return a + b

    def echo(value):
        log.append(["echo", {"value": value}])
        return value

    def boom(message):
        log.append(["boom", {"message": message}])
        raise RuntimeError(message)

    def bad(message):
        log.append(["bad", {"message": message}])
        raise ValueError(message)

    return {fn.__name__: fn for fn in (fetch, words, count, add, echo, boom, bad)}


def _drive(plan, tool_names, max_steps):
    log = []
    tools = {name: _fixtures(log)[name] for name in tool_names}
    try:
        trace = _execute(plan, tools, max_steps)
    except ValueError:
        return {"error": "ValueError", "calls": log}
    return {"trace": trace, "calls": log}


def reference(plan, tools, max_steps):
    return _drive(plan, tools, max_steps)


def call(module, args):
    if not hasattr(module, "solve"):
        raise AttributeError("the solution does not define solve()")
    plan, tool_names, max_steps = args
    log = []
    tools = {name: _fixtures(log)[name] for name in tool_names}
    try:
        trace = module.solve(plan, tools, max_steps)
    except ValueError:
        return {"error": "ValueError", "calls": log}
    return {"trace": trace, "calls": log}


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, dict):
        raise AssertionError(f"internal result should be a dict, got {type(got).__name__}: {got!r:.200}")
    if "error" in want:
        if got.get("error") != "ValueError":
            raise AssertionError(
                f"expected ValueError before any step ran, got {got!r:.400}\n"
                "  invalid plans (too long, a duplicate id, or a reference that is not an earlier step)"
                " must raise before the first tool call"
            )
        if got.get("calls"):
            raise AssertionError(
                f"validation failed, so no tool should run, but these calls happened: {got.get('calls')!r:.400}"
            )
        return
    if got.get("error"):
        raise AssertionError(
            f"the plan is valid, but solve raised ValueError. Tool failures belong in the trace.\n"
            f"  calls so far: {got.get('calls')!r:.300}"
        )
    try:
        default_compare(got.get("calls"), want["calls"], 1e-6, 1e-6, "tool calls")
    except AssertionError as exc:
        raise AssertionError(
            f"tool calls differ (skipped steps and unknown tools must not run; a tool that raises still ran)\n  {exc}"
        ) from None
    try:
        default_compare(got.get("trace"), want["trace"], 1e-6, 1e-6, "trace")
    except AssertionError as exc:
        raise AssertionError(f"trace differs\n  {exc}") from None


def _step(step_id, tool, args):
    return {"id": step_id, "tool": tool, "args": args}


def _random(seed, n, name):
    rng = np.random.default_rng(seed)
    plan = []
    for i in range(n):
        roll = float(rng.random())
        if i == 0 or roll < 0.4:
            tool, args = "echo", {"value": int(rng.integers(0, 40))}
        elif roll < 0.65:
            src = int(rng.integers(0, i))
            tool, args = "echo", {"value": {"n": f"$s{src}.output", "tag": "keep"}}
        elif roll < 0.85:
            src = int(rng.integers(0, i))
            tool, args = "add", {"a": f"$s{src}.output", "b": int(rng.integers(0, 5))}
        else:
            tool, args = "boom", {"message": f"e{i}"}
        if float(rng.random()) < 0.07:
            tool = "missing"
        plan.append(_step(f"s{i}", tool, args))
    return {"_name": name, "plan": plan, "tools": ["echo", "add", "boom"], "max_steps": n}


def functional_cases():
    echo = ["echo"]
    return [
        {"_name": "example", "tools": ["fetch", "words", "count", "add"], "max_steps": 6, "plan": [
            _step("s1", "fetch", {"url": "a.com"}),
            _step("s2", "words", {"text": "$s1.output"}),
            _step("s3", "fetch", {"url": "c.com"}),
            _step("s4", "count", {"items": "$s3.output"}),
            _step("s5", "count", {"items": "$s2.output"}),
            _step("s6", "add", {"a": "$s5.output", "b": "$s4.output"})]},
        {"_name": "empty plan", "plan": [], "tools": echo, "max_steps": 0},
        {"_name": "over the step budget runs nothing", "tools": echo, "max_steps": 1, "plan": [
            _step("s1", "echo", {"value": 1}),
            _step("s2", "echo", {"value": 2})]},
        {"_name": "duplicate id runs nothing", "tools": echo, "max_steps": 5, "plan": [
            _step("s1", "echo", {"value": 1}),
            _step("s1", "echo", {"value": 2})]},
        {"_name": "forward reference runs nothing", "tools": echo, "max_steps": 5, "plan": [
            _step("s1", "echo", {"value": 1}),
            _step("s2", "echo", {"value": "$s3.output"}),
            _step("s3", "echo", {"value": 2})]},
        {"_name": "self reference runs nothing", "tools": echo, "max_steps": 3, "plan": [
            _step("s1", "echo", {"value": ["$s1.output"]})]},
        {"_name": "unknown reference runs nothing", "tools": echo, "max_steps": 4, "plan": [
            _step("s1", "echo", {"value": "ok"}),
            _step("s2", "echo", {"value": "$missing.output"})]},
        {"_name": "only an exact reference is substituted", "tools": ["echo", "boom"], "max_steps": 5, "plan": [
            _step("z", "boom", {"message": "down"}),
            _step("lit", "echo", {"value": "total: $z.output"}),
            _step("real", "echo", {"value": "$z.output"}),
            _step("pad", "echo", {"value": "$z.output "}),
            _step("mix", "echo", {"value": {"n": 2, "half": 1.5, "ok": False, "missing": None,
                                            "note": "total: $z.output"}})]},
        {"_name": "nested values are substituted, dict keys are not", "tools": echo, "max_steps": 2, "plan": [
            _step("s1", "echo", {"value": {"k": "v", "n": 1}}),
            _step("s2", "echo", {"value": {
                "$s1.output": "key-is-literal",
                "nested": ["$s1.output", {"inner": "$s1.output"}, "total: $s1.output"],
                "flag": True}})]},
        {"_name": "unknown tool does not block an independent step", "tools": echo, "max_steps": 5, "plan": [
            _step("s1", "echo", {"value": 1}),
            _step("s2", "missing", {}),
            _step("s3", "echo", {"value": "$s2.output"}),
            _step("s4", "echo", {"value": "$s1.output"}),
            _step("s5", "echo", {"value": "after"})]},
        {"_name": "blockers are sorted, unique, and propagate", "tools": ["echo", "boom"], "max_steps": 4, "plan": [
            _step("z", "boom", {"message": "x"}),
            _step("m", "echo", {"value": "$z.output"}),
            _step("c", "echo", {"value": {"p": "$z.output", "q": "$m.output", "r": "$z.output"}}),
            _step("d", "echo", {"value": "still"})]},
        {"_name": "a tool ValueError stays in the trace", "tools": ["bad", "echo"], "max_steps": 2, "plan": [
            _step("s1", "bad", {"message": "nope"}),
            _step("s2", "echo", {"value": "$s1.output"})]},
        _random(1, 12, "random 12 steps"),
        _random(2, 30, "random 30 steps"),
    ]


def build() -> Problem:
    return Problem(
        slug="planner_executor",
        title="Planner/executor boundary",
        difficulty="medium",
        summary="Execute a bounded tool plan with step references, skip what depends on failures, and return a structured trace.",
        params=["plan", "tools", "max_steps"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        atol=1e-6,
        rtol=1e-6,
        tags=["agents", "planning", "orchestration"],
    )
