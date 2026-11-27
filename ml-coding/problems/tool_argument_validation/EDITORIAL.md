#### The idea
Model-proposed arguments are untrusted input, so validate them against the schema before anything runs. The core is a
function that maps a Python value to its **JSON** type name. The ordering matters: test `bool` before `int`, because in
Python `True` is an `int`. Then each schema type accepts a set of JSON types (`number` accepts `integer` and `number`).

Loop over the union of required names and supplied names in sorted order, and give each name its first error only.

#### A solution
```python
import json


def json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


ACCEPTS = {
    "string": {"string"},
    "integer": {"integer"},
    "number": {"integer", "number"},
    "boolean": {"boolean"},
    "array": {"array"},
}


def in_enum(value, options):
    return any(value == option and isinstance(value, bool) == isinstance(option, bool) for option in options)


def solve(schema, args):
    if not isinstance(args, dict):
        return [f"arguments: expected object, got {json_type(args)}"]
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    strict = schema.get("additionalProperties", True) is False
    errors = []
    for name in sorted(required | set(args)):
        if name not in args:
            errors.append(f"missing required argument: {name}")
        elif name not in properties:
            if strict:
                errors.append(f"unexpected argument: {name}")
        else:
            spec, value = properties[name], args[name]
            if "type" in spec and json_type(value) not in ACCEPTS[spec["type"]]:
                errors.append(f"{name}: expected {spec['type']}, got {json_type(value)}")
            elif "enum" in spec and not in_enum(value, spec["enum"]):
                errors.append(f"{name}: {json.dumps(value)} is not one of {json.dumps(spec['enum'])}")
    return errors
```

#### What it costs
`O(A log A)` for `A` argument names. The errors are worth sending back to the model verbatim: a precise message such as
`days: expected integer, got boolean` lets it fix the call on the next turn, which is how structured-output retries
work in agent frameworks.

#### Pitfalls
- `isinstance(True, int)` is `True`, so a naive check accepts booleans as integers.
- Rejecting `3` for a `number` (JSON numbers include integers), or accepting `3.0` for an `integer`.
- `True in [1, 2]` is `True` in Python; the enum check has to compare types too.
- Reporting several errors for one name, or listing them in the order they were found instead of sorted.
