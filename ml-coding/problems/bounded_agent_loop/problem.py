"""Grader for bounded_agent_loop. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Cases are JSON. `tools` is a list of fixture names. The script is consumed as written;
fixtures log each real call, including calls that then raise.
"""

import numpy as np

from gymkit.spec import Problem

_KEYS = {"answer", "stop_reason", "trace", "calls"}


def _run(script, tools, max_steps, loop_limit):
    trace = []
    index = 0
    steps = 0
    previous = None
    streak = 0
    while True:
        if steps >= max_steps:
            return {"answer": None, "stop_reason": "step_budget", "trace": trace}
        if index >= len(script):
            return {"answer": None, "stop_reason": "model_exhausted", "trace": trace}
        response = script[index]
        index += 1
        steps += 1
        trace.append({"kind": "model", "response": response})
        if response["type"] == "final":
            return {"answer": response["answer"], "stop_reason": "final", "trace": trace}
        tool, args = response["tool"], response["args"]
        proposal = (tool, args)
        if proposal == previous:
            streak += 1
        else:
            previous = proposal
            streak = 1
        if streak >= loop_limit:
            return {"answer": None, "stop_reason": "loop_detected", "trace": trace}
        if tool not in tools:
            trace.append({
                "kind": "observation",
                "tool": tool,
                "ok": False,
                "error": f"unknown tool: {tool}",
            })
            continue
        try:
            output = tools[tool](**args)
        except Exception as exc:
            trace.append({
                "kind": "observation",
                "tool": tool,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
        else:
            trace.append({"kind": "observation", "tool": tool, "ok": True, "output": output})


def _fixtures(log):
    def search(query):
        log.append(["search", {"query": query}])
        return [query]

    def add(a, b):
        log.append(["add", {"a": a, "b": b}])
        return a + b

    def echo(value):
        log.append(["echo", {"value": value}])
        return value

    def fail(message):
        log.append(["fail", {"message": message}])
        raise RuntimeError(message)

    return {fn.__name__: fn for fn in (search, add, echo, fail)}


def _drive(solve, script, tool_names, max_steps, loop_limit):
    log = []
    tools = {name: _fixtures(log)[name] for name in tool_names}
    result = solve(script, tools, max_steps, loop_limit)
    if not isinstance(result, dict):
        return result
    out = dict(result)
    out["calls"] = log
    return out


def reference(script, tools, max_steps, loop_limit):
    return _drive(_run, script, tools, max_steps, loop_limit)


def call(module, args):
    if not hasattr(module, "solve"):
        raise AttributeError("the solution does not define solve()")
    return _drive(module.solve, *args)


def compare(got, want, case):
    if not isinstance(got, dict) or set(got) != _KEYS:
        keys = sorted(got) if isinstance(got, dict) else type(got).__name__
        raise AssertionError(f"return keys {keys}; expected answer, stop_reason, trace, calls")
    if got["answer"] != want["answer"] or got["stop_reason"] != want["stop_reason"]:
        raise AssertionError(
            f"answer {got['answer']!r}, stop_reason {got['stop_reason']!r}\n"
            f"  expected answer {want['answer']!r}, stop_reason {want['stop_reason']!r}"
        )
    if got["calls"] != want["calls"]:
        raise AssertionError(
            "tool calls differ (an unknown tool is not called, and the proposal that trips the loop is not called)\n"
            f"  got:      {got['calls']!r:.500}\n  expected: {want['calls']!r:.500}"
        )
    g, w = got["trace"], want["trace"]
    if not isinstance(g, list) or len(g) != len(w):
        length = len(g) if isinstance(g, list) else type(g).__name__
        raise AssertionError(f"trace length {length}, expected {len(w)}\n  got: {g!r:.400}")
    for index, (a, b) in enumerate(zip(g, w)):
        if a != b:
            raise AssertionError(f"trace[{index}]:\n  got:      {a!r:.400}\n  expected: {b!r:.400}")


def _tool(name, args):
    return {"type": "tool", "tool": name, "args": args}


def _final(answer):
    return {"type": "final", "answer": answer}


_CATALOG = ["search", "add", "echo", "fail"]


def _random(seed, n, name):
    rng = np.random.default_rng(seed)
    script = []
    for i in range(n):
        if script and float(rng.random()) < 0.3 and script[-1]["type"] == "tool":
            prev = script[-1]
            script.append(_tool(prev["tool"], dict(prev["args"])))
            continue
        if float(rng.random()) < 0.12:
            script.append(_final(f"done-{i}"))
            continue
        tool = _CATALOG[int(rng.integers(0, len(_CATALOG)))]
        if float(rng.random()) < 0.1:
            tool = "missing"
        if tool == "add":
            args = {"a": int(rng.integers(0, 6)), "b": int(rng.integers(0, 6))}
        elif tool == "echo":
            args = {"value": int(rng.integers(0, 6))}
        elif tool == "fail":
            args = {"message": f"m{i}"}
        else:
            args = {"query": f"q{int(rng.integers(0, 4))}"}
        script.append(_tool(tool, args))
    return {
        "_name": name,
        "script": script,
        "tools": list(_CATALOG),
        "max_steps": int(rng.integers(1, n + 2)),
        "loop_limit": int(rng.integers(2, 5)),
    }


def functional_cases():
    tools = list(_CATALOG)
    return [
        {"_name": "example", "tools": tools, "max_steps": 4, "loop_limit": 3, "script": [
            _tool("search", {"query": "gpu"}),
            _tool("missing", {}),
            _final(2)]},
        {"_name": "final answer can be null", "tools": tools, "max_steps": 2, "loop_limit": 2, "script": [
            _final(None)]},
        {"_name": "the step budget leaves the rest of the script", "tools": tools, "max_steps": 1,
         "loop_limit": 3, "script": [
             _tool("search", {"query": "a"}),
             _tool("search", {"query": "b"}),
             _final("nope")]},
        {"_name": "a budget of zero consumes nothing", "tools": tools, "max_steps": 0, "loop_limit": 2,
         "script": [_final("nope")]},
        {"_name": "the third identical call is not executed", "tools": tools, "max_steps": 8,
         "loop_limit": 3, "script": [
             _tool("search", {"query": "gpu"}),
             _tool("search", {"query": "gpu"}),
             _tool("search", {"query": "gpu"}),
             _final("no")]},
        {"_name": "dict key order does not make a new call", "tools": tools, "max_steps": 5,
         "loop_limit": 2, "script": [
             _tool("add", {"a": 1, "b": 2}),
             _tool("add", {"b": 2, "a": 1}),
             _final("nope")]},
        {"_name": "a different call resets the streak", "tools": tools, "max_steps": 6, "loop_limit": 3,
         "script": [
             _tool("search", {"query": "a"}),
             _tool("search", {"query": "a"}),
             _tool("search", {"query": "b"}),
             _tool("search", {"query": "b"}),
             _final("ok")]},
        {"_name": "unknown tool on the last step does not consume the next response", "tools": tools,
         "max_steps": 1, "loop_limit": 3, "script": [
             _tool("missing", {}),
             _final("nope")]},
        {"_name": "a tool exception is an observation and the loop continues", "tools": tools,
         "max_steps": 3, "loop_limit": 3, "script": [
             _tool("fail", {"message": "boom"}),
             _final("after")]},
        {"_name": "a repeated unknown tool trips the loop", "tools": tools, "max_steps": 5,
         "loop_limit": 2, "script": [
             _tool("missing", {}),
             _tool("missing", {}),
             _final("no")]},
        {"_name": "the script runs out after a tool call", "tools": tools, "max_steps": 5,
         "loop_limit": 3, "script": [
             _tool("echo", {"value": 1})]},
        {"_name": "a final answer on the last allowed step wins", "tools": tools, "max_steps": 1,
         "loop_limit": 2, "script": [
             _final({"ok": True}),
             _tool("search", {"query": "a"})]},
        _random(1, 18, "random 18 responses"),
        _random(2, 40, "random 40 responses"),
    ]


def build() -> Problem:
    return Problem(
        slug="bounded_agent_loop",
        title="Bounded agent tool-use loop",
        difficulty="hard",
        summary="Run a scripted tool-use loop that stops on a final answer, a step budget, or a repeated call, and return the trace.",
        params=["script", "tools", "max_steps", "loop_limit"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        atol=1e-6,
        rtol=1e-6,
        tags=["agents", "tools", "termination"],
    )
