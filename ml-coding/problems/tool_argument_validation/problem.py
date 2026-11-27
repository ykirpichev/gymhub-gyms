"""Grader for tool_argument_validation. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import json

import numpy as np

from gymkit.spec import Problem

_ACCEPTS = {"string": {"string"}, "integer": {"integer"}, "number": {"integer", "number"},
            "boolean": {"boolean"}, "array": {"array"}}


def _json_type(value):
    for kind, name in ((type(None), "null"), (bool, "boolean"), (int, "integer"), (float, "number"),
                       (str, "string"), (list, "array"), (dict, "object")):
        if isinstance(value, kind):
            return name
    return type(value).__name__


def reference(schema, args):
    if not isinstance(args, dict):
        return [f"arguments: expected object, got {_json_type(args)}"]
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    strict = schema.get("additionalProperties", True) is False
    errors = []
    for name in sorted(required | set(args)):
        if name not in args:
            errors.append(f"missing required argument: {name}")
            continue
        if name not in props:
            if strict:
                errors.append(f"unexpected argument: {name}")
            continue
        spec, value = props[name], args[name]
        if "type" in spec and _json_type(value) not in _ACCEPTS[spec["type"]]:
            errors.append(f"{name}: expected {spec['type']}, got {_json_type(value)}")
        elif "enum" in spec and not any(value == e and isinstance(value, bool) == isinstance(e, bool) for e in spec["enum"]):
            errors.append(f"{name}: {json.dumps(value)} is not one of {json.dumps(spec['enum'])}")
    return errors


def compare(got, want, case):
    if not isinstance(got, list):
        raise AssertionError(f"return a list of error strings, got {type(got).__name__}: {got!r:.120}")
    for index, (g, w) in enumerate(zip(got, want)):
        if g != w:
            raise AssertionError(f"error #{index}: got {g!r}, expected {w!r}\n  got:      {got!r}\n  expected: {want!r}")
    if len(got) != len(want):
        extra = got[len(want):] or want[len(got):]
        what = "extra" if len(got) > len(want) else "missing"
        raise AssertionError(f"{len(got)} errors, expected {len(want)}; {what}: {extra!r}\n  expected: {want!r}")


def _schema(properties, required=(), strict=None):
    schema = {"type": "object", "properties": properties, "required": list(required)}
    if strict is not None:
        schema["additionalProperties"] = not strict
    return schema


_EXAMPLE = _schema({"city": {"type": "string"}, "days": {"type": "integer"}, "units": {"enum": ["C", "F"]}},
                   ["city", "days"], strict=True)

_TYPES = ["string", "integer", "number", "boolean", "array"]
_VALUES = [None, True, False, 0, 1, 7, -3, 2.5, 3.0, "", "x", "C", [], [1, 2], {"k": 1}]


def _random(seed, n_props, name):
    rng = np.random.default_rng(seed)
    props = {}
    for i in range(n_props):
        roll = int(rng.integers(3))
        spec = {}
        if roll != 1:
            spec["type"] = _TYPES[int(rng.integers(len(_TYPES)))]
        if roll != 0:
            picks = rng.choice(len(_VALUES), size=3, replace=False)
            spec["enum"] = [_VALUES[int(j)] for j in picks]
        props[f"p{i:02d}"] = spec
    names = list(props)
    required = [n for n in names if rng.random() < 0.4]
    args = {}
    for n in names + [f"extra{i}" for i in range(3)]:
        if rng.random() < 0.7:
            spec = props.get(n, {})
            if "enum" in spec and rng.random() < 0.5:
                args[n] = spec["enum"][int(rng.integers(len(spec["enum"])))]
            else:
                args[n] = _VALUES[int(rng.integers(len(_VALUES)))]
    return {"_name": name, "schema": _schema(props, required, strict=bool(rng.random() < 0.5)), "args": args}


def functional_cases():
    numbers = _schema({"n": {"type": "integer"}, "x": {"type": "number"}, "flag": {"type": "boolean"}},
                      ["n", "x", "flag"])
    return [
        {"_name": "example", "schema": _EXAMPLE, "args": {"days": True, "units": "K", "lang": "en"}},
        {"_name": "valid call", "schema": _EXAMPLE, "args": {"city": "Oslo", "days": 3, "units": "F"}},
        {"_name": "no arguments, nothing required", "schema": _schema({"q": {"type": "string"}}), "args": {}},
        {"_name": "bool is not an integer or a number", "schema": numbers, "args": {"n": False, "x": True, "flag": 1}},
        {"_name": "integer is a number, 3.0 is not an integer", "schema": numbers, "args": {"n": 3.0, "x": 3, "flag": True}},
        {"_name": "missing required, sorted by name", "schema": _schema(
            {"b": {"type": "string"}, "a": {"type": "string"}, "c": {"type": "array"}}, ["c", "b", "a"]), "args": {}},
        {"_name": "unexpected arguments rejected when strict", "schema": _schema(
            {"q": {"type": "string"}}, ["q"], strict=True), "args": {"q": "hi", "Z": 1, "a": None}},
        {"_name": "extra arguments ignored by default", "schema": _schema(
            {"q": {"type": "string"}}, ["q"]), "args": {"q": "hi", "verbose": True}},
        {"_name": "enum: wrong value, true is not 1", "schema": _schema(
            {"level": {"enum": [1, 2, 3]}, "mode": {"type": "string", "enum": ["fast", "slow"]},
             "tag": {"enum": [None, "x"]}}),
         "args": {"level": True, "mode": "medium", "tag": None}},
        {"_name": "type is checked before enum", "schema": _schema({"mode": {"type": "string", "enum": ["a", "b"]}}),
         "args": {"mode": 1}},
        {"_name": "null and object values", "schema": _schema(
            {"s": {"type": "string"}, "xs": {"type": "array"}}, ["s"]), "args": {"s": None, "xs": {"0": 1}}},
        {"_name": "arguments are not an object", "schema": _EXAMPLE, "args": ["Oslo", 3]},
        _random(1, 8, "random 8 properties"),
        _random(2, 40, "random 40 properties"),
    ]


def build() -> Problem:
    return Problem(
        slug="tool_argument_validation",
        title="Validate JSON tool arguments",
        difficulty="easy",
        summary="Check a model's tool-call arguments against a small JSON Schema subset and report every error in a fixed order.",
        params=["schema", "args"],
        reference=reference,
        functional_cases=functional_cases,
        compare=compare,
        tags=["agents", "tools", "validation"],
    )
