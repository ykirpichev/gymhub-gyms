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
