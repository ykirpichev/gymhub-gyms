"""Grader for tool_registry. Private: learners see PROBLEM.md, HINTS.md and the starter only.

Cases are JSON: ops are ["register", name, impl, schema, side_effect] or ["call", name, args, approved].
`impl` names one of the fixture functions below; each run gets fresh fixtures that log their side effects."""

import inspect

import numpy as np

from gymkit.spec import Problem

_ACCEPTS = {"string": {"string"}, "integer": {"integer"}, "number": {"integer", "number"},
            "boolean": {"boolean"}, "array": {"array"}, "object": {"object"}}


def _json_type(value):
    for kind, name in ((type(None), "null"), (bool, "boolean"), (int, "integer"), (float, "number"),
                       (str, "string"), (list, "array"), (dict, "object")):
        if isinstance(value, kind):
            return name
    return type(value).__name__


class _Registry:
    def __init__(self):
        self.tools = {}

    def register(self, name, fn, schema, side_effect=False):
        params = inspect.signature(fn).parameters
        required = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        if (name in self.tools or any(k not in params or t not in _ACCEPTS for k, t in schema.items())
                or not required <= set(schema)):
            raise ValueError(name)
        self.tools[name] = (fn, schema, required, side_effect)

    def call(self, name, args, approved=False):
        if name not in self.tools:
            return {"ok": False, "error": "unknown_tool"}
        fn, schema, required, side_effect = self.tools[name]
        details = []
        for k in sorted(set(args) | required):
            if k not in args:
                details.append(f"missing required argument: {k}")
            elif k not in schema:
                details.append(f"unexpected argument: {k}")
            elif _json_type(args[k]) not in _ACCEPTS[schema[k]]:
                details.append(f"{k}: expected {schema[k]}, got {_json_type(args[k])}")
        if details:
            return {"ok": False, "error": "invalid_arguments", "details": details}
        if side_effect and not approved:
            return {"ok": False, "error": "approval_required"}
        try:
            return {"ok": True, "result": fn(**args)}
        except Exception as exc:
            return {"ok": False, "error": "tool_error", "message": f"{type(exc).__name__}: {exc}"}


def _fixtures(log):
    files = {"a.txt", "notes.md"}

    def add(a, b=0):
        return a + b

    def search(query, limit=3):
        return [f"{query}-{i}" for i in range(limit)]

    def send_email(to, subject, body=""):
        log.append(["send_email", to, subject])
        return {"id": sum(1 for entry in log if entry[0] == "send_email")}

    def divide(x, y):
        return x / y

    def delete_file(path):
        if path not in files:
            raise FileNotFoundError(f"no such file: {path}")
        files.remove(path)
        log.append(["delete_file", path])
        return True

    def tag(item, labels, meta=None):
        return {"item": item, "labels": labels, "meta": meta}

    return {f.__name__: f for f in (add, search, send_email, divide, delete_file, tag)}


def _drive(cls, ops):
    log = []
    tools = _fixtures(log)
    registry = cls()
    results = []
    for op in ops:
        if op[0] == "register":
            _, name, impl, schema, side_effect = op
            try:
                registry.register(name, tools[impl], schema, side_effect=side_effect)
                results.append("registered")
            except ValueError:
                results.append("ValueError")
        elif op[0] == "call":
            _, name, args, approved = op
            results.append(registry.call(name, args, approved=approved))
        else:
            raise ValueError(f"unknown op {op[0]!r}")
    return {"results": results, "side_effects": log}


def reference(ops):
    return _drive(_Registry, ops)


def call(module, args):
    if not hasattr(module, "ToolRegistry"):
        raise AttributeError("the solution does not define class ToolRegistry")
    return _drive(module.ToolRegistry, *args)


def compare(got, want, case):
    from gymkit.runner import default_compare

    ops = case["ops"]
    for index, (g, w) in enumerate(zip(got["results"], want["results"])):
        try:
            default_compare(g, w, 1e-9, 1e-9, "result")
        except AssertionError:
            what = "register" if ops[index][0] == "register" else "call"
            raise AssertionError(f"ops[{index}] = {ops[index]!r:.140}\n  {what} gave {g!r:.200}\n  expected   {w!r:.200}") from None
    if got["side_effects"] != want["side_effects"]:
        raise AssertionError(f"side effects performed by the tools: {got['side_effects']!r:.200}\n"
                             f"  expected: {want['side_effects']!r:.200}")


_ADD = ["register", "add", "add", {"a": "integer", "b": "integer"}, False]
_EMAIL = ["register", "send_email", "send_email", {"to": "string", "subject": "string", "body": "string"}, True]
_DIV = ["register", "divide", "divide", {"x": "number", "y": "number"}, False]
_DEL = ["register", "delete_file", "delete_file", {"path": "string"}, True]
_TAG = ["register", "tag", "tag", {"item": "string", "labels": "array", "meta": "object"}, False]
_SEARCH = ["register", "search", "search", {"query": "string", "limit": "integer"}, False]

_ARG_POOL = [None, True, 0, 2, -1, 1.5, "", "a.txt", "x", [], ["red"], {"k": 1}]
_TYPED = {"string": ["a.txt", "notes.md", "x", ""], "integer": [0, 2, -1], "number": [0, 2, 1.5],
          "array": [[], ["red"]], "object": [{}, {"k": 1}]}


def _random(seed, n_ops, name):
    rng = np.random.default_rng(seed)
    schemas = {op[1]: op[3] for op in (_ADD, _EMAIL, _DIV, _DEL, _TAG, _SEARCH)}
    ops = [_ADD, _EMAIL, _DIV, _DEL, _TAG, _SEARCH]
    for _ in range(n_ops):
        tool = list(schemas)[int(rng.integers(len(schemas)))] if rng.random() < 0.95 else "missing_tool"
        args = {}
        for arg in schemas.get(tool, {"q": "string"}):
            roll = rng.random()
            if roll < 0.75:
                pool = _TYPED.get(schemas.get(tool, {}).get(arg), _ARG_POOL)
                args[arg] = pool[int(rng.integers(len(pool)))]
            elif roll < 0.9:
                args[arg] = _ARG_POOL[int(rng.integers(len(_ARG_POOL)))]
        if rng.random() < 0.1:
            args["extra"] = 1
        ops.append(["call", tool, args, bool(rng.random() < 0.5)])
    return {"_name": name, "ops": ops}


def functional_cases():
    email = {"to": "ann@example.com", "subject": "hi"}
    return [
        {"_name": "example", "ops": [
            _ADD, _EMAIL,
            ["call", "add", {"a": 2, "b": 3}, False],
            ["call", "add", {"a": "2", "c": 1}, False],
            ["call", "send_email", email, False],
            ["call", "send_email", email, True],
            ["call", "rm", {}, False]]},
        {"_name": "defaults fill omitted arguments", "ops": [
            _ADD, _SEARCH,
            ["call", "add", {"a": 4}, False],
            ["call", "search", {"query": "gpu"}, False],
            ["call", "search", {"query": "gpu", "limit": 1}, False]]},
        {"_name": "bool is not an integer, int is a number", "ops": [
            _ADD, _DIV,
            ["call", "add", {"a": True, "b": 1}, False],
            ["call", "divide", {"x": 7, "y": 2}, False],
            ["call", "divide", {"x": 1.5, "y": False}, False]]},
        {"_name": "missing and unexpected, sorted by name", "ops": [
            _EMAIL,
            ["call", "send_email", {"zzz": 1, "body": 5, "aaa": None}, True]]},
        {"_name": "invalid arguments are reported before approval", "ops": [
            _EMAIL,
            ["call", "send_email", {"to": "bob@example.com"}, False]]},
        {"_name": "no side effect without approval", "ops": [
            _DEL, _EMAIL,
            ["call", "delete_file", {"path": "a.txt"}, False],
            ["call", "send_email", email, False],
            ["call", "delete_file", {"path": "a.txt"}, True],
            ["call", "send_email", email, True],
            ["call", "send_email", email, True]]},
        {"_name": "tool exceptions become results", "ops": [
            _DIV, _DEL,
            ["call", "divide", {"x": 1, "y": 0}, False],
            ["call", "delete_file", {"path": "missing.txt"}, True],
            ["call", "delete_file", {"path": "notes.md"}, True],
            ["call", "delete_file", {"path": "notes.md"}, True]]},
        {"_name": "approval on a read-only tool is harmless", "ops": [
            _ADD, ["call", "add", {"a": 1, "b": 2}, True]]},
        {"_name": "array and object arguments", "ops": [
            _TAG,
            ["call", "tag", {"item": "car", "labels": ["red", "fast"]}, False],
            ["call", "tag", {"item": "car", "labels": "red", "meta": []}, False],
            ["call", "tag", {"item": "car", "labels": [], "meta": {"src": "user"}}, False]]},
        {"_name": "register rejects bad schemas", "ops": [
            _ADD,
            ["register", "add", "search", {"query": "string"}, False],
            ["register", "add2", "add", {"a": "integer", "c": "integer"}, False],
            ["register", "add3", "add", {"b": "integer"}, False],
            ["register", "add4", "add", {"a": "int"}, False],
            ["register", "add5", "add", {"a": "number"}, False],
            ["call", "add2", {"a": 1}, False],
            ["call", "add5", {"a": 1.5}, False],
            ["call", "add5", {"a": 1, "b": 2}, False]]},
        _random(1, 30, "random 30 calls"),
        _random(2, 400, "random 400 calls"),
    ]


def build() -> Problem:
    return Problem(
        slug="tool_registry",
        title="Permission-aware tool registry",
        difficulty="medium",
        summary="Register Python functions as tools, validate and bind model arguments, and refuse side effects without approval.",
        params=["ops"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["agents", "tools", "permissions", "class design"],
    )
